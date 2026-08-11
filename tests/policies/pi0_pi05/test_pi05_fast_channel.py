#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for piR2's fast proprioception channel on pi0.5 (arXiv 2607.26055, Sec. 3.2).

Upstream pi0.5 discretizes state into the tokenized prompt, which traps proprioception inside the
cacheable vision-language prefix. ``state_in_suffix`` moves it into the action expert's suffix (as
pi0 does) so it can be refreshed on every denoising step. These tests pin the suffix layout, the
staleness conditioning, and that the default configuration is untouched.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.utils.constants import OBS_STATE
from tests.utils import require_cuda

CHUNK_SIZE = 8
STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SIZE = 224
DEVICE = "cuda"


def _config(**kwargs) -> PI05Config:
    """Smallest pi0.5 that still exercises the real suffix layout."""
    config = PI05Config(
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        dtype="float32",
        **kwargs,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)
        ),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return config


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_state_stays_in_the_prompt_by_default():
    config = _config()
    assert config.state_in_suffix is False
    assert config.vlm_delay_max == 0


def test_delay_conditioning_requires_the_fast_channel():
    # A stale prefix is only usable if proprioception can still be refreshed behind it.
    with pytest.raises(ValueError, match="requires state_in_suffix=True"):
        _config(vlm_delay_max=4)


def test_negative_delay_budget_is_rejected():
    with pytest.raises(ValueError, match="vlm_delay_max must be >= 0"):
        _config(state_in_suffix=True, vlm_delay_max=-1)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _run_tokenizer_step(include_state_in_prompt: bool) -> str:
    step = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=STATE_DIM, include_state_in_prompt=include_state_in_prompt
    )
    transition: EnvTransition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, STATE_DIM)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up the cube"]},
    }
    return step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]


def test_prompt_carries_discretized_state_by_default():
    prompt = _run_tokenizer_step(include_state_in_prompt=True)
    assert "State:" in prompt
    assert prompt.startswith("Task: pick up the cube, State: ")


def test_fast_channel_prompt_drops_state_but_keeps_the_cleaned_task():
    prompt = _run_tokenizer_step(include_state_in_prompt=False)
    assert "State:" not in prompt
    # Underscore and newline cleaning must survive the shortcut.
    assert prompt == "Task: pick up the cube;\nAction: "


def test_processor_flag_follows_the_policy_config():
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    for state_in_suffix in (False, True):
        preprocessor, _ = make_pi05_pre_post_processors(
            _config(state_in_suffix=state_in_suffix),
            dataset_stats=None,
        )
        step = next(s for s in preprocessor.steps if isinstance(s, Pi05PrepareStateTokenizerProcessorStep))
        assert step.include_state_in_prompt is not state_in_suffix


# ---------------------------------------------------------------------------
# Suffix layout (needs a real model)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompt_state_model():
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    return PI05Pytorch(_config()).to(DEVICE).eval()


@pytest.fixture(scope="module")
def fast_channel_model():
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    return PI05Pytorch(_config(state_in_suffix=True, vlm_delay_max=4)).to(DEVICE).eval()


def _suffix_inputs(batch=2):
    x_t = torch.randn(batch, CHUNK_SIZE, ACTION_DIM, device=DEVICE)
    timestep = torch.rand(batch, device=DEVICE)
    return x_t, timestep


def _state(batch=2, value=0.0):
    return torch.full((batch, STATE_DIM), value, device=DEVICE)


@require_cuda
def test_prompt_state_model_has_no_fast_channel_weights(prompt_state_model):
    assert not hasattr(prompt_state_model, "state_proj")
    assert not hasattr(prompt_state_model, "vlm_delay_proj")


@require_cuda
def test_prompt_state_suffix_is_actions_only(prompt_state_model):
    x_t, timestep = _suffix_inputs()
    embs, pad_masks, att_masks, cond = prompt_state_model.embed_suffix(x_t, timestep)

    assert embs.shape[1] == CHUNK_SIZE
    assert pad_masks.shape[1] == CHUNK_SIZE
    assert att_masks.shape[1] == CHUNK_SIZE
    assert cond.ndim == 2


@require_cuda
def test_passing_state_to_a_prompt_state_model_is_an_error(prompt_state_model):
    x_t, timestep = _suffix_inputs()
    with pytest.raises(ValueError, match="state_in_suffix=False"):
        prompt_state_model.embed_suffix(x_t, timestep, state=_state())


@require_cuda
def test_fast_channel_prepends_exactly_one_state_token(fast_channel_model):
    x_t, timestep = _suffix_inputs()
    embs, pad_masks, att_masks, _ = fast_channel_model.embed_suffix(x_t, timestep, state=_state())

    assert embs.shape[1] == CHUNK_SIZE + 1
    assert pad_masks.shape[1] == CHUNK_SIZE + 1
    # The state token opens its own attention block, then the action tokens attend to it.
    assert att_masks[0].tolist() == [1.0, 1.0] + [0.0] * (CHUNK_SIZE - 1)


@require_cuda
def test_state_actually_changes_the_suffix_embedding(fast_channel_model):
    x_t, timestep = _suffix_inputs(batch=1)
    low = fast_channel_model.embed_suffix(x_t, timestep, state=_state(batch=1, value=0.0))[0]
    high = fast_channel_model.embed_suffix(x_t, timestep, state=_state(batch=1, value=1.0))[0]

    # Only the state token may move; the action tokens do not depend on it at embedding time.
    assert not torch.allclose(low[:, 0], high[:, 0])
    torch.testing.assert_close(low[:, 1:], high[:, 1:])


@require_cuda
def test_state_token_shares_the_first_positions_per_token_conditioning(fast_channel_model):
    x_t, _ = _suffix_inputs(batch=1)
    # A staircase schedule gives every position its own timestep.
    position_time = torch.linspace(0, 1, CHUNK_SIZE, device=DEVICE)[None, :]
    _, _, _, cond = fast_channel_model.embed_suffix(x_t, position_time, state=_state(batch=1))

    assert cond.shape[1] == CHUNK_SIZE + 1
    torch.testing.assert_close(cond[:, 0], cond[:, 1])


@require_cuda
def test_prefix_staleness_changes_the_conditioning(fast_channel_model):
    x_t, timestep = _suffix_inputs(batch=1)
    state = _state(batch=1)
    fresh = fast_channel_model.embed_suffix(
        x_t, timestep, state=state, vlm_delay=torch.zeros(1, dtype=torch.long, device=DEVICE)
    )[3]
    stale = fast_channel_model.embed_suffix(
        x_t, timestep, state=state, vlm_delay=torch.full((1,), 4, dtype=torch.long, device=DEVICE)
    )[3]

    assert not torch.allclose(fresh, stale)


@require_cuda
def test_training_forward_runs_with_the_full_pir2_configuration(fast_channel_model):
    """The staircase, the state token and the delay embedding all at once, end to end.

    This is the combination a piR2 fine-tune actually uses, and it is where a shape mismatch
    between per-position conditioning and the extra suffix token would surface. Runs
    language-only: the vision tower is unrelated to the suffix layout, and the small VLM variant
    that keeps this test cheap hardcodes an incompatible image projection width.
    """
    from lerobot.policies.pi05.modeling_pi05 import _build_staircase_schedule

    batch, delay = 2, 2
    images, img_masks = [], []
    tokens = torch.randint(0, 1000, (batch, 20), device=DEVICE)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    actions = torch.randn(batch, CHUNK_SIZE, ACTION_DIM, device=DEVICE)
    noise = torch.randn_like(actions)
    time = torch.rand(batch, device=DEVICE)
    prefix_mask, position_time = _build_staircase_schedule(batch, CHUNK_SIZE, delay, time)

    losses = fast_channel_model.forward(
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask,
        position_time,
        state=torch.randn(batch, STATE_DIM, device=DEVICE),
        vlm_delay=torch.randint(0, 5, (batch,), device=DEVICE),
    )

    assert losses.shape == (batch, CHUNK_SIZE, ACTION_DIM)
    assert torch.isfinite(losses).all()


@require_cuda
def test_delay_is_ignored_when_not_provided(fast_channel_model):
    x_t, timestep = _suffix_inputs(batch=1)
    state = _state(batch=1)
    without = fast_channel_model.embed_suffix(x_t, timestep, state=state)[3]
    zero = fast_channel_model.embed_suffix(
        x_t, timestep, state=state, vlm_delay=torch.zeros(1, dtype=torch.long, device=DEVICE)
    )[3]

    # A delay of zero is still a learned offset, so it is not the same as omitting the channel.
    assert not torch.allclose(without, zero)

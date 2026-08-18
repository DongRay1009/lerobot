"""Ground-truth scene memory for LIBERO Spatial task 5.

Phase 2 starts with MuJoCo state so that failure-detection logic can be
validated independently of perception. A later phase can replace this state
provider with vision estimates without changing the detector interface.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np


TASK5_BODIES = {
    "eef": "gripper0_eef",
    "target_bowl": "akita_black_bowl_1_main",
    "distractor_bowl": "akita_black_bowl_2_main",
    "ramekin": "glazed_rim_porcelain_ramekin_1_main",
    "plate": "plate_1_main",
}


def find_sim(root):
    queue = [root]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        sim = getattr(obj, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return sim

        for attr in ("env", "_env", "venv", "_venv", "unwrapped"):
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None and child is not obj:
                queue.append(child)

        children = getattr(obj, "envs", None)
        if isinstance(children, dict):
            queue.extend(list(children.values()))
        elif children:
            queue.extend(list(children))

    raise RuntimeError("MuJoCo simulator not found in environment wrappers")


def body_id(model, name):
    lookup = getattr(model, "body_name2id", None)
    if callable(lookup):
        return int(lookup(name))
    native = getattr(model, "_model", model)
    result = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise KeyError(f"MuJoCo body not found: {name}")
    return int(result)


def body_position(sim, body_index):
    positions = getattr(sim.data, "body_xpos", None)
    if positions is None:
        positions = sim.data.xpos
    return np.asarray(positions[body_index], dtype=float).copy()


class Task5SceneMemory:
    def __init__(self, env):
        self.env = env
        self.sim = None
        self.ids = {}
        self._history = deque(maxlen=6)
        self.initial_bowl_z = None
        self.ever_held = False
        self.was_over_plate = False
        self.placement_failure_count = 0
        self.reset()

    def reset(self):
        # LIBERO uses hard_reset=True by default, which rebuilds MjSim between
        # episodes. Refresh both the simulator reference and body ids.
        self.sim = find_sim(self.env)
        self.ids = {
            key: body_id(self.sim.model, name)
            for key, name in TASK5_BODIES.items()
        }
        self._history.clear()
        self.ever_held = False
        self.was_over_plate = False
        self.placement_failure_count = 0
        self.initial_bowl_z = float(self._position("target_bowl")[2])

    def _position(self, key):
        return body_position(self.sim, self.ids[key])

    def update(self):
        positions = {key: self._position(key) for key in self.ids}
        eef = positions["eef"]
        bowl = positions["target_bowl"]
        ramekin = positions["ramekin"]
        plate = positions["plate"]

        eef_bowl_distance = float(np.linalg.norm(eef - bowl))
        bowl_ramekin_xy = float(np.linalg.norm((bowl - ramekin)[:2]))
        bowl_plate_xy = float(np.linalg.norm((bowl - plate)[:2]))
        bowl_plate_z = float(bowl[2] - plate[2])
        bowl_lift = float(bowl[2] - self.initial_bowl_z)

        self._history.append({"eef": eef.copy(), "bowl": bowl.copy()})
        follows_eef = False
        if len(self._history) == self._history.maxlen:
            eef_delta = self._history[-1]["eef"] - self._history[0]["eef"]
            bowl_delta = self._history[-1]["bowl"] - self._history[0]["bowl"]
            follows_eef = bool(
                np.linalg.norm(bowl_delta) > 0.005
                and np.linalg.norm(bowl_delta - eef_delta) < 0.025
                and eef_bowl_distance < 0.12
            )

        near_bowl = eef_bowl_distance < 0.075
        lifted = bowl_lift > 0.025
        held = follows_eef and lifted
        self.ever_held = self.ever_held or held

        over_plate = bowl_plate_xy < 0.05 and lifted
        self.was_over_plate = self.was_over_plate or over_plate
        settled_at_plate_height = -0.005 < bowl_plate_z < 0.035

        # Calibrated from the two successful and one failed task-5 traces:
        # success settled within 1.6/3.0 cm of the plate center, while the
        # failed placement settled about 5.0 cm away.
        on_plate = (
            bowl_plate_xy < 0.035
            and settled_at_plate_height
            and not held
        )
        placement_failed_raw = (
            self.ever_held
            and self.was_over_plate
            and not held
            and settled_at_plate_height
            and bowl_plate_xy >= 0.035
        )
        if placement_failed_raw:
            self.placement_failure_count += 1
        else:
            self.placement_failure_count = 0
        placement_failed = self.placement_failure_count >= 5
        on_ramekin = bowl_ramekin_xy < 0.075 and not lifted

        if on_plate:
            stage = "complete"
        elif placement_failed:
            stage = "placement_failed"
        elif self.was_over_plate and not held:
            stage = "verify_place"
        elif held and over_plate:
            stage = "place"
        elif held:
            stage = "transport"
        elif lifted:
            stage = "lift"
        elif near_bowl:
            stage = "grasp"
        else:
            stage = "approach"

        return {
            "positions": {
                key: value.round(6).tolist() for key, value in positions.items()
            },
            "eef_bowl_distance": eef_bowl_distance,
            "bowl_ramekin_xy_distance": bowl_ramekin_xy,
            "bowl_plate_xy_distance": bowl_plate_xy,
            "bowl_plate_z_distance": bowl_plate_z,
            "bowl_lift": bowl_lift,
            "near_bowl": near_bowl,
            "bowl_follows_eef": follows_eef,
            "held": held,
            "ever_held": self.ever_held,
            "over_plate": over_plate,
            "placement_failed_raw": placement_failed_raw,
            "placement_failure_count": self.placement_failure_count,
            "placement_failed": placement_failed,
            "on_ramekin": on_ramekin,
            "on_plate": on_plate,
            "stage": stage,
        }

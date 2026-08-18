"""Trace a LeRobot LIBERO evaluation run.

Run this file on the Ubuntu host from the lerobot conda environment. It
monkey-patches the LIBERO environment factory, then forwards all normal
lerobot-eval arguments while saving per-step actions, rewards and states.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import numpy as np

from scene_memory import Task5SceneMemory
from placement_recovery import Task5PlacementRecovery
from failure_detector import Task5FailureDetector
from vision_scene_memory import VisionTask5SceneMemory


OUT = Path(os.environ.get("TRACE_OUT", "./task5_trace"))
OUT.mkdir(parents=True, exist_ok=True)


def _summary(value):
    """Convert nested numpy observations into compact JSON metadata."""
    if isinstance(value, dict):
        return {str(k): _summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary(v) for v in value]
    arr = np.asarray(value)
    if arr.dtype.kind in "biufc":
        # Keep state/action arrays verbatim, but do not expand camera frames
        # into enormous JSON lists. Videos already preserve visual evidence.
        if arr.size > 256:
            return {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        flat = arr.astype(float).reshape(-1)
        return {
            "shape": list(arr.shape),
            "values": flat.tolist(),
        }
    return {"type": type(value).__name__, "shape": list(arr.shape)}


class TraceEnv:
    def __init__(self, env, name):
        self._env = env
        self._name = name
        self._episode = -1
        self._step = 0
        self._file = None
        self._last_obs = None
        self._pose_recovery_phase = None
        self._pose_recovery_step = 0
        self._first_close_seen = False
        self._pose_target = None
        self._scene_memory = None
        self._last_scene = None
        self._vision_scene_memory = None
        self._last_vision_scene = None
        self._placement_recovery = Task5PlacementRecovery()
        self._failure_detector = Task5FailureDetector()
        self._dir = OUT / name
        self._dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, key):
        return getattr(self._env, key)

    def reset(self, *args, **kwargs):
        if self._file is not None:
            self._file.close()
        self._episode += 1
        self._step = 0
        self._file = (self._dir / f"episode_{self._episode}.jsonl").open("w")
        result = self._env.reset(*args, **kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        if self._name.endswith("_5"):
            if self._scene_memory is None:
                self._scene_memory = Task5SceneMemory(self._env)
            else:
                self._scene_memory.reset()
            self._last_scene = self._scene_memory.update()
            vision_enabled = (
                os.environ.get("VISION_SHADOW", "0") == "1"
                or os.environ.get("VISION_DETECTOR", "0") == "1"
                or os.environ.get("VISION_RECOVERY_TARGETS", "0") == "1"
            )
            if vision_enabled:
                if self._vision_scene_memory is None:
                    self._vision_scene_memory = VisionTask5SceneMemory(self._env)
                else:
                    self._vision_scene_memory.reset()
                self._last_vision_scene = self._vision_scene_memory.update(obs)
            self._placement_recovery.reset()
            self._failure_detector.reset()
        self._last_obs = obs
        self._pose_recovery_phase = None
        self._pose_recovery_step = 0
        self._first_close_seen = False
        self._pose_target = None
        self._file.write(json.dumps({"event": "reset", "obs": _summary(obs)}) + "\n")
        self._file.flush()
        return result

    def step(self, action):
        original_action = np.array(action, copy=True)
        phase_executed = None
        # Minimal intervention experiment: if a rollout is still running
        # after 100 steps while the policy keeps the gripper open, force a
        # close command for this step. This is deliberately a diagnostic
        # heuristic, not the final recovery controller.
        recovery_applied = False
        failure = {
            "grasp_pending": False,
            "steps_since_close": 0,
            "max_lift_since_close": None,
            "grasp_failed": False,
        }
        use_vision_detector = os.environ.get("VISION_DETECTOR", "0") == "1"
        use_vision_recovery_targets = (
            os.environ.get("VISION_RECOVERY_TARGETS", "0") == "1"
        )
        detector_scene = (
            self._last_vision_scene
            if use_vision_detector
            else self._last_scene
        )
        if not self._placement_recovery.active:
            failure = self._failure_detector.update(
                original_action, detector_scene
            )
        if os.environ.get("PLACE_RECOVERY", "0") == "1":
            recovery_scene = self._last_scene
            external_trigger = failure["grasp_failed"]
            if use_vision_detector and detector_scene is not None:
                # Detection comes exclusively from vision in this hybrid
                # experiment. Keep GT positions only as recovery targets.
                external_trigger = (
                    external_trigger
                    or detector_scene.get("placement_failed", False)
                )
                recovery_scene = dict(self._last_scene)
                recovery_scene["placement_failed"] = False
            if use_vision_recovery_targets and detector_scene is not None:
                # Full oracle-mask RGB-D control experiment: both the trigger
                # and recovery object targets come from vision. EEF remains
                # ordinary robot state, as it would through ROS / joint state.
                recovery_scene = dict(detector_scene)
                recovery_scene["placement_failed"] = False
            override, place_phase = self._placement_recovery.compute(
                original_action,
                recovery_scene,
                external_trigger=external_trigger,
            )
            if override is not None:
                action = override
                recovery_applied = True
                phase_executed = f"place_{place_phase}"
        if os.environ.get("FORCE_GRIPPER_RECOVERY", "0") == "1":
            action_arr = np.asarray(action)
            if self._step >= 100 and action_arr[..., -1].mean() < -0.8:
                action = np.array(action_arr, copy=True)
                action[..., -1] = 1.0
                recovery_applied = True

        # Pose-aware retry for LIBERO Spatial task 5. Successful traces close
        # near [-0.223, 0.142, 0.980]; the reproducible failure closes too far
        # toward -x and too low. When the first close command arrives from the
        # bad region, briefly re-align with an open gripper, then close and
        # lift before returning control to the policy.
        if os.environ.get("POSE_GRASP_RECOVERY", "0") == "1":
            policy_action = np.asarray(original_action)
            if not self._first_close_seen and policy_action[..., -1].mean() > 0.8:
                self._first_close_seen = True
                try:
                    eef = np.asarray(self._last_obs["robot_state"]["eef"]["pos"])[0]
                    if eef[0] < -0.245 or eef[2] < 0.970:
                        self._pose_recovery_phase = "align"
                        self._pose_recovery_step = 0
                        # Mean successful first-close pose from episodes 0/1.
                        self._pose_target = np.array([-0.2235, 0.1417, 0.9796])
                except (KeyError, IndexError, TypeError):
                    pass

            if self._pose_recovery_phase is not None:
                action = np.zeros_like(policy_action)
                recovery_applied = True
                phase_executed = self._pose_recovery_phase

                try:
                    eef = np.asarray(self._last_obs["robot_state"]["eef"]["pos"])[0]
                except (KeyError, IndexError, TypeError):
                    eef = self._pose_target

                if self._pose_recovery_phase == "align":
                    error = self._pose_target - eef
                    action[..., :3] = np.clip(8.0 * error, -0.4, 0.4)
                    action[..., -1] = -1.0
                    self._pose_recovery_step += 1
                    if np.linalg.norm(error) < 0.006 or self._pose_recovery_step >= 20:
                        self._pose_recovery_phase = "close"
                        self._pose_recovery_step = 0
                elif self._pose_recovery_phase == "close":
                    error = self._pose_target - eef
                    action[..., :3] = np.clip(6.0 * error, -0.3, 0.3)
                    action[..., -1] = 1.0
                    self._pose_recovery_step += 1
                    if self._pose_recovery_step >= 12:
                        self._pose_recovery_phase = "lift"
                        self._pose_recovery_step = 0
                elif self._pose_recovery_phase == "lift":
                    lift_target = self._pose_target + np.array([0.0, 0.0, 0.05])
                    error = lift_target - eef
                    action[..., :3] = np.clip(6.0 * error, -0.3, 0.3)
                    action[..., -1] = 1.0
                    self._pose_recovery_step += 1
                    if np.linalg.norm(error) < 0.008 or self._pose_recovery_step >= 15:
                        self._pose_recovery_phase = None
                        self._pose_recovery_step = 0
            else:
                phase_executed = None

        result = self._env.step(action)
        obs, reward, terminated, truncated, info = result
        self._last_obs = obs
        scene = self._scene_memory.update() if self._scene_memory is not None else None
        vision_scene = (
            self._vision_scene_memory.update(obs)
            if self._vision_scene_memory is not None
            else None
        )
        self._last_scene = scene
        self._last_vision_scene = vision_scene
        row = {
            "event": "step",
            "step": self._step,
            "policy_action": _summary(original_action),
            "action": _summary(action),
            "recovery_applied": recovery_applied,
            "recovery_phase": phase_executed,
            "detector_source": (
                "vision_scene" if use_vision_detector else "scene"
            ),
            "recovery_target_source": (
                "vision_scene" if use_vision_recovery_targets else "scene"
            ),
            "reward": _summary(reward),
            "terminated": _summary(terminated),
            "truncated": _summary(truncated),
            "info": _summary(info),
            "obs": _summary(obs),
            "scene": scene,
            "vision_scene": vision_scene,
            "failure": failure,
        }
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        self._step += 1
        return result

    def close(self):
        if self._file is not None:
            self._file.close()
        return self._env.close()


def _patch_factory():
    import lerobot.envs as envs_module

    original = envs_module.make_env

    def traced_make_env(*args, **kwargs):
        envs = original(*args, **kwargs)
        for suite, suite_envs in envs.items():
            # LeRobot 0.6 returns a dict keyed by task id (older versions may
            # return a list).  Iterate over a snapshot so replacing values
            # does not mutate the dictionary while it is being traversed.
            if isinstance(suite_envs, dict):
                for key, env in list(suite_envs.items()):
                    suite_envs[key] = TraceEnv(env, f"{suite}_{key}")
            else:
                for index, env in enumerate(list(suite_envs)):
                    suite_envs[index] = TraceEnv(env, f"{suite}_{index}")
        return envs

    envs_module.make_env = traced_make_env


if __name__ == "__main__":
    _patch_factory()
    # lerobot-eval's normal command-line arguments are preserved.
    runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")

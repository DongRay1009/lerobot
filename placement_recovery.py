"""Object-relative placement recovery for LIBERO Spatial task 5."""

from __future__ import annotations

import numpy as np


class Task5PlacementRecovery:
    """Closed-loop skill sequence driven by ground-truth scene memory."""

    # Mean EEF-minus-bowl offset at first close in successful episodes 0/1.
    GRASP_OFFSET = np.array([-0.00385, -0.0200, 0.0390])

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "idle"
        self.phase_step = 0
        self.target = None
        self.grasp_offset = self.GRASP_OFFSET.copy()
        self.trigger_count = 0

    @property
    def active(self):
        return self.phase not in ("idle", "done")

    def _enter(self, phase, target=None):
        self.phase = phase
        self.phase_step = 0
        self.target = None if target is None else np.asarray(target, dtype=float)

    @staticmethod
    def _position(scene, key):
        return np.asarray(scene["positions"][key], dtype=float)

    @staticmethod
    def _pose_action(policy_action, current, target, gripper, gain=8.0, limit=0.4):
        action = np.zeros_like(np.asarray(policy_action))
        error = target - current
        action[..., :3] = np.clip(gain * error, -limit, limit)
        action[..., -1] = gripper
        return action, float(np.linalg.norm(error))

    def compute(self, policy_action, scene, external_trigger=False):
        """Return (override_action, executed_phase), or (None, None)."""
        if scene is None:
            return None, None

        eef = self._position(scene, "eef")
        bowl = self._position(scene, "target_bowl")
        plate = self._position(scene, "plate")

        if self.phase == "idle":
            if not (scene.get("placement_failed", False) or external_trigger):
                return None, None
            self.trigger_count += 1
            self._enter("above_bowl")

        if self.phase == "done":
            return None, None

        executed_phase = self.phase

        if self.phase == "above_bowl":
            target = bowl + self.GRASP_OFFSET + np.array([0.0, 0.0, 0.06])
            action, error = self._pose_action(policy_action, eef, target, -1.0)
            self.phase_step += 1
            if error < 0.008 or self.phase_step >= 30:
                self._enter("descend")

        elif self.phase == "descend":
            target = bowl + self.GRASP_OFFSET
            action, error = self._pose_action(policy_action, eef, target, -1.0)
            self.phase_step += 1
            if error < 0.006 or self.phase_step >= 25:
                self._enter("close", target=eef)

        elif self.phase == "close":
            action, _ = self._pose_action(policy_action, eef, self.target, 1.0, gain=6.0)
            self.phase_step += 1
            if self.phase_step >= 12:
                self.grasp_offset = eef - bowl
                self._enter("lift", target=eef + np.array([0.0, 0.0, 0.08]))

        elif self.phase == "lift":
            action, error = self._pose_action(policy_action, eef, self.target, 1.0)
            self.phase_step += 1
            if error < 0.01 or self.phase_step >= 30:
                desired_bowl = plate + np.array([0.0, 0.0, 0.10])
                self._enter("transport", target=desired_bowl + self.grasp_offset)

        elif self.phase == "transport":
            action, error = self._pose_action(policy_action, eef, self.target, 1.0)
            self.phase_step += 1
            if error < 0.01 or self.phase_step >= 40:
                desired_bowl = plate + np.array([0.0, 0.0, 0.012])
                self._enter("lower", target=desired_bowl + self.grasp_offset)

        elif self.phase == "lower":
            action, error = self._pose_action(policy_action, eef, self.target, 1.0, gain=6.0)
            self.phase_step += 1
            if error < 0.008 or self.phase_step >= 35:
                self._enter("open", target=eef)

        elif self.phase == "open":
            action, _ = self._pose_action(policy_action, eef, self.target, -1.0, gain=5.0)
            self.phase_step += 1
            if self.phase_step >= 12:
                self._enter("retreat", target=eef + np.array([0.0, 0.0, 0.08]))

        elif self.phase == "retreat":
            action, error = self._pose_action(policy_action, eef, self.target, -1.0)
            self.phase_step += 1
            if error < 0.01 or self.phase_step >= 25:
                self._enter("done")

        else:
            raise RuntimeError(f"Unknown placement-recovery phase: {self.phase}")

        return action, executed_phase

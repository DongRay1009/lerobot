"""Temporal failure detectors for LIBERO Spatial task 5."""

from __future__ import annotations

import numpy as np


class Task5FailureDetector:
    """Detect a failed grasp after observing the post-close outcome."""

    def __init__(self, verification_steps=25, min_lift=0.015):
        self.verification_steps = verification_steps
        self.min_lift = min_lift
        self.reset()

    def reset(self):
        self.previous_gripper_command = -1.0
        self.pending_grasp = False
        self.steps_since_close = 0
        self.max_lift_since_close = float("-inf")
        self.reported = False

    def update(self, policy_action, scene):
        result = {
            "grasp_pending": self.pending_grasp,
            "steps_since_close": self.steps_since_close,
            "max_lift_since_close": None,
            "grasp_failed": False,
        }
        if scene is None:
            return result

        action = np.asarray(policy_action)
        gripper_command = float(action[..., -1].mean())
        close_started = (
            gripper_command > 0.8
            and self.previous_gripper_command <= 0.8
            and not scene.get("ever_held", False)
        )
        self.previous_gripper_command = gripper_command

        if close_started and not self.reported:
            self.pending_grasp = True
            self.steps_since_close = 0
            self.max_lift_since_close = float(scene["bowl_lift"])

        if self.pending_grasp:
            self.steps_since_close += 1
            self.max_lift_since_close = max(
                self.max_lift_since_close, float(scene["bowl_lift"])
            )

            if scene.get("held", False) or self.max_lift_since_close >= self.min_lift:
                self.pending_grasp = False
            elif self.steps_since_close >= self.verification_steps:
                result["grasp_failed"] = True
                self.pending_grasp = False
                self.reported = True

        result.update(
            {
                "grasp_pending": self.pending_grasp,
                "steps_since_close": self.steps_since_close,
                "max_lift_since_close": (
                    None
                    if self.max_lift_since_close == float("-inf")
                    else self.max_lift_since_close
                ),
            }
        )
        return result

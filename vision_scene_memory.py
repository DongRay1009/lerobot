"""RGB-D scene memory for LIBERO Spatial task 5.

The first perception milestone uses MuJoCo instance segmentation as oracle
masks, but estimates positions from agentview depth and camera calibration.
It never reads object body positions. This lets us validate the geometry and
state-tracking interface before replacing masks with a learned model.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from inspect_task5_backprojection import (
    backproject_mask,
    camera_calibration,
    estimate,
)
from inspect_task5_masks import OBJECT_KEYS, geom_ids_for_body_tree
from inspect_task5_vision import depth_to_meters, robosuite_segmentation
from scene_memory import TASK5_BODIES, body_id, find_sim


CAMERA = "agentview"
WIDTH = 360
HEIGHT = 360
MIN_PIXELS = 50
STATIC_LANDMARKS = {"ramekin"}
PLATE_RELOCALIZATION_MIN_PIXELS = 1000
PLATE_RELOCALIZATION_MIN_VISIBLE_FRACTION = 0.80
PLATE_RELOCALIZATION_DISTANCE = 0.015
PLATE_RELOCALIZATION_CONFIRMATION_STEPS = 5
PLATE_RELOCALIZATION_MAX_SPREAD = 0.008
ON_PLATE_CONFIRMATION_STEPS = 3
ON_PLATE_DISTANCE = 0.032
PLACEMENT_FAILURE_DISTANCE = 0.035
PLACEMENT_FAILURE_CONFIRMATION_STEPS = 8

# Fixed surface-median -> body-origin corrections measured once during RGB-D
# calibration. The same bowl correction is shared by both identical bowls.
POSITION_CORRECTIONS = {
    "target_bowl": np.array([0.01833, -0.00525, -0.03058]),
    "distractor_bowl": np.array([0.01833, -0.00525, -0.03058]),
    "ramekin": np.array([-0.02113, -0.00602, -0.02014]),
    "plate": np.array([0.00152, -0.00189, -0.00712]),
}


def eef_position_from_obs(obs):
    value = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=float)
    return value.reshape(-1, 3)[0].copy()


class VisionTask5SceneMemory:
    """Task-5 scene state estimated from RGB-D plus oracle instance masks."""

    def __init__(self, env):
        self.env = env
        self.sim = None
        self.root_ids = {}
        self.geom_ids = {}
        self.target_instance = None
        self.initial_bowl_z = None
        self.last_positions = {}
        self._history = deque(maxlen=6)
        self.ever_held = False
        self.was_over_plate = False
        self.on_plate_count = 0
        self.placement_failure_count = 0
        self._plate_candidates = deque(
            maxlen=PLATE_RELOCALIZATION_CONFIRMATION_STEPS
        )
        self.plate_reference_pixels = None
        self.plate_relocalization_count = 0
        self.reset()

    def reset(self):
        self.sim = find_sim(self.env)
        self.root_ids = {
            key: body_id(self.sim.model, TASK5_BODIES[key])
            for key in OBJECT_KEYS
        }
        self.geom_ids = {
            key: geom_ids_for_body_tree(self.sim.model, root_id)
            for key, root_id in self.root_ids.items()
        }
        self.target_instance = None
        self.initial_bowl_z = None
        self.last_positions = {}
        self._history.clear()
        self.ever_held = False
        self.was_over_plate = False
        self.on_plate_count = 0
        self.placement_failure_count = 0
        self._plate_candidates.clear()
        self.plate_reference_pixels = None
        self.plate_relocalization_count = 0

    def _plate_position(self, measured, pixel_count):
        """Keep a stable plate anchor, but accept persistent physical motion."""
        if "plate" not in self.last_positions:
            self.last_positions["plate"] = measured.copy()
            self.plate_reference_pixels = pixel_count
            return measured, False, 0.0

        anchor = self.last_positions["plate"]
        displacement = float(np.linalg.norm((measured - anchor)[:2]))
        visible_fraction = pixel_count / max(self.plate_reference_pixels, 1)
        confident_change = (
            pixel_count >= PLATE_RELOCALIZATION_MIN_PIXELS
            and visible_fraction >= PLATE_RELOCALIZATION_MIN_VISIBLE_FRACTION
            and displacement >= PLATE_RELOCALIZATION_DISTANCE
        )
        if not confident_change:
            self._plate_candidates.clear()
            return anchor.copy(), False, displacement

        self._plate_candidates.append(measured.copy())
        if len(self._plate_candidates) < self._plate_candidates.maxlen:
            return anchor.copy(), False, displacement

        candidates = np.asarray(self._plate_candidates)
        candidate = np.median(candidates, axis=0)
        spread = float(
            np.max(np.linalg.norm(candidates[:, :2] - candidate[:2], axis=1))
        )
        if spread > PLATE_RELOCALIZATION_MAX_SPREAD:
            return anchor.copy(), False, displacement

        self.last_positions["plate"] = candidate.copy()
        self._plate_candidates.clear()
        self.plate_relocalization_count += 1
        return candidate, True, displacement

    def _observe_objects(self):
        _, raw_depth = self.sim.render(
            camera_name=CAMERA,
            width=WIDTH,
            height=HEIGHT,
            depth=True,
        )
        depth, _, _ = depth_to_meters(self.sim, raw_depth)
        segmentation = robosuite_segmentation(self.sim, CAMERA)
        intrinsics, camera_position, camera_rotation = camera_calibration(
            self.sim, CAMERA, WIDTH, HEIGHT
        )
        geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)

        positions = {}
        observations = {}
        for key in OBJECT_KEYS:
            mask = (
                (segmentation[:, :, 0] == geom_objtype)
                & np.isin(segmentation[:, :, 1], self.geom_ids[key])
            )
            pixel_count = int(mask.sum())
            position = None
            relocalized = False
            anchor_displacement = None
            if pixel_count >= MIN_PIXELS:
                points = backproject_mask(
                    mask,
                    depth,
                    intrinsics,
                    camera_position,
                    camera_rotation,
                    y_sign=1.0,
                )
                surface_position = estimate(points)
                if surface_position is not None:
                    measured = surface_position + POSITION_CORRECTIONS[key]
                    # Ramekin stays fixed. Plate is normally anchored too, but
                    # persistent, well-observed displacement indicates that a
                    # collision physically moved it and updates the anchor.
                    if key == "plate":
                        position, relocalized, anchor_displacement = (
                            self._plate_position(measured, pixel_count)
                        )
                    elif key in STATIC_LANDMARKS and key in self.last_positions:
                        position = self.last_positions[key].copy()
                    else:
                        position = measured
                        self.last_positions[key] = position.copy()

            stale = position is None
            if stale and key in self.last_positions:
                position = self.last_positions[key].copy()
            if position is not None:
                positions[key] = position
            observations[key] = {
                "visible": not stale,
                "stale": stale,
                "anchored": (
                    key in STATIC_LANDMARKS or key == "plate"
                ) and key in self.last_positions,
                "pixel_count": pixel_count,
                "visible_fraction": (
                    pixel_count / max(self.plate_reference_pixels, 1)
                    if key == "plate" and self.plate_reference_pixels is not None
                    else None
                ),
                "relocalized": relocalized,
                "anchor_displacement": anchor_displacement,
                "relocalization_candidate_count": (
                    len(self._plate_candidates) if key == "plate" else 0
                ),
            }
        return positions, observations

    def update(self, obs):
        observed, observations = self._observe_objects()
        required = {"target_bowl", "distractor_bowl", "ramekin", "plate"}
        if not required.issubset(observed):
            missing = sorted(required - set(observed))
            raise RuntimeError(f"Vision scene memory missing initial objects: {missing}")

        if self.target_instance is None:
            ramekin = observed["ramekin"]
            bowl_keys = ("target_bowl", "distractor_bowl")
            self.target_instance = min(
                bowl_keys,
                key=lambda key: np.linalg.norm((observed[key] - ramekin)[:2]),
            )
            self.initial_bowl_z = float(observed[self.target_instance][2])

        other_instance = (
            "distractor_bowl"
            if self.target_instance == "target_bowl"
            else "target_bowl"
        )
        positions = {
            "eef": eef_position_from_obs(obs),
            "target_bowl": observed[self.target_instance],
            "distractor_bowl": observed[other_instance],
            "ramekin": observed["ramekin"],
            "plate": observed["plate"],
        }

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
                and np.linalg.norm(bowl_delta - eef_delta) < 0.035
                and eef_bowl_distance < 0.13
            )

        near_bowl = eef_bowl_distance < 0.085
        lifted = bowl_lift > 0.025
        held = follows_eef and lifted
        self.ever_held = self.ever_held or held
        over_plate = bowl_plate_xy < 0.055 and lifted
        self.was_over_plate = self.was_over_plate or over_plate
        settled_at_plate_height = -0.01 < bowl_plate_z < 0.045
        on_plate_raw = (
            bowl_plate_xy < ON_PLATE_DISTANCE
            and settled_at_plate_height
            and not held
        )
        self.on_plate_count = self.on_plate_count + 1 if on_plate_raw else 0
        on_plate = self.on_plate_count >= ON_PLATE_CONFIRMATION_STEPS

        placement_failed_raw = (
            self.ever_held
            and self.was_over_plate
            and not held
            and settled_at_plate_height
            and bowl_plate_xy > PLACEMENT_FAILURE_DISTANCE
        )
        self.placement_failure_count = (
            self.placement_failure_count + 1 if placement_failed_raw else 0
        )
        placement_failed = (
            self.placement_failure_count
            >= PLACEMENT_FAILURE_CONFIRMATION_STEPS
        )
        on_ramekin = bowl_ramekin_xy < 0.085 and not lifted

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
            "source": "agentview_rgbd_oracle_masks",
            "target_instance": self.target_instance,
            "positions": {
                key: value.round(6).tolist() for key, value in positions.items()
            },
            "observations": observations,
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
            "plate_relocalization_count": self.plate_relocalization_count,
            "on_ramekin": on_ramekin,
            "on_plate_raw": on_plate_raw,
            "on_plate_count": self.on_plate_count,
            "on_plate": on_plate,
            "stage": stage,
        }

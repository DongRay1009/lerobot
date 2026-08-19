"""Task-5 scene memory driven by Grounding DINO, SAM, and RGB-D."""

from __future__ import annotations

import itertools
import os

import numpy as np

from inspect_task5_backprojection import (
    backproject_mask,
    camera_calibration,
    estimate,
)
from inspect_task5_vision import depth_to_meters
from learned_mask_perception import Task5LearnedMaskPerception
from scene_memory import find_sim
from vision_scene_memory import (
    CAMERA,
    HEIGHT,
    MIN_PIXELS,
    POSITION_CORRECTIONS,
    PLATE_RELOCALIZATION_CONFIRMATION_STEPS,
    STATIC_LANDMARKS,
    WIDTH,
    VisionTask5SceneMemory,
    eef_position_from_obs,
)


OBJECT_KEYS = ("target_bowl", "distractor_bowl", "ramekin", "plate")
GRIPPER_ATTACH_APERTURE = 0.055
GRIPPER_RELEASE_APERTURE = 0.065
TARGET_ATTACH_DISTANCE = 0.10
DISTRACTOR_ASSOCIATION_DISTANCE = 0.08
TARGET_ASSOCIATION_DISTANCE = 0.12
LEARNED_ON_PLATE_CONFIRMATION_STEPS = 2
PLACEMENT_VERIFICATION_TIMEOUT_STEPS = 15
PLACEMENT_RELEASE_XY_LIMIT = 0.18
TARGET_MEASUREMENT_INNOVATION_LIMIT = 0.10
DISTRACTOR_MEASUREMENT_INNOVATION_LIMIT = 0.06
INITIALIZATION_CLASS_THRESHOLDS = {
    "black bowl": 0.08,
    "white plate": 0.20,
    "white ramekin": 0.15,
}
INITIALIZATION_BOX_THRESHOLD = 0.05
INITIALIZATION_TEXT_THRESHOLD = 0.15
EYE_CAMERA = "robot0_eye_in_hand"
# Eye-camera surface median -> bowl origin calibration from the initial
# target-on-ramekin view. The eye camera is used only to complete reset-time
# bowl initialization when agentview misses one instance.
EYE_BOWL_POSITION_CORRECTION = np.array([0.00010, -0.01978, -0.02430])
# When the bowl is held, the wrist camera observes a different visible
# surface than at reset. This correction is calibrated from held-view RGB-D
# measurements, separately from the reset-time target-on-ramekin view.
EYE_HELD_BOWL_POSITION_CORRECTION = np.array(
    [-0.0215, -0.0004, -0.0170]
)
EYE_HELD_MIN_DETECTION_SCORE = 0.25
EYE_HELD_TARGET_INNOVATION_LIMIT = 0.04
MULTIVIEW_DEDUPLICATION_DISTANCE = 0.06
STATIC_WORKSPACE_MIN = np.array([-0.50, 0.05, 0.84])
STATIC_WORKSPACE_MAX = np.array([0.40, 0.40, 1.02])
RAMEKIN_FROM_TARGET_OFFSET = np.array([0.0145, 0.0343, -0.0430])


class LearnedVisionTask5SceneMemory(VisionTask5SceneMemory):
    """Drop-in visual scene memory using learned masks instead of segmentation."""

    def __init__(self, env, perception_interval=None):
        self.perception_interval = perception_interval or int(
            os.environ.get("LEARNED_VISION_INTERVAL", "5")
        )
        self.perception = Task5LearnedMaskPerception()
        self._current_obs = None
        self._frame_index = 0
        self._last_timing = None
        self._last_detection_counts = {}
        self._last_seen_frame = {}
        self._last_eef = None
        self._target_attached = False
        self._placement_attempted = False
        self._placement_verification_count = 0
        self._initialization_fallback_used = False
        self._initialization_eye_used = False
        self._initialization_eye_counts = {}
        self._ramekin_relational_anchor = False
        super().__init__(env)

    def reset(self):
        # Reset task memory without constructing oracle body/geometry mappings.
        self.sim = find_sim(self.env)
        self.root_ids = {}
        self.geom_ids = {}
        self.target_instance = None
        self.initial_bowl_z = None
        self.last_positions = {}
        self._history.clear()
        self.ever_held = False
        self.was_over_plate = False
        self.on_plate_count = 0
        self.placement_failure_count = 0
        self._plate_candidates.clear()
        self._plate_candidates = type(self._plate_candidates)(
            maxlen=PLATE_RELOCALIZATION_CONFIRMATION_STEPS
        )
        self.plate_reference_pixels = None
        self.plate_relocalization_count = 0
        self._current_obs = None
        self._frame_index = 0
        self._last_timing = None
        self._last_detection_counts = {}
        self._last_seen_frame = {}
        self._last_eef = None
        self._target_attached = False
        self._placement_attempted = False
        self._placement_verification_count = 0
        self._initialization_fallback_used = False
        self._initialization_eye_used = False
        self._initialization_eye_counts = {}
        self._ramekin_relational_anchor = False

    def _held_observations(self, perception_frame=False):
        observations = {}
        for key in OBJECT_KEYS:
            observations[key] = {
                "visible": False,
                "stale": True,
                "anchored": key in STATIC_LANDMARKS or key == "plate",
                "pixel_count": 0,
                "detection_score": None,
                "perception_frame": perception_frame,
                "inference_timing_ms": self._last_timing,
                "detection_counts": self._last_detection_counts,
                "last_seen_frame": self._last_seen_frame.get(key),
                "propagated": False,
                "measurement_rejected": False,
                "measurement_innovation": None,
                "camera": None,
                "workspace_rejected": False,
                "relational_fallback": False,
            }
        return {
            key: value.copy() for key, value in self.last_positions.items()
        }, observations

    def _fresh_observation(self, detection):
        return {
            "visible": True,
            "stale": False,
            "anchored": False,
            "pixel_count": int(detection["mask"].sum()),
            "detection_score": float(detection["score"]),
            "sam_predicted_iou": float(detection["sam_predicted_iou"]),
            "perception_frame": True,
            "inference_timing_ms": self._last_timing,
            "detection_counts": self._last_detection_counts,
            "last_seen_frame": self._frame_index,
            "propagated": False,
            "measurement_rejected": False,
            "measurement_innovation": 0.0,
            "camera": detection.get("camera", CAMERA),
            "workspace_rejected": False,
            "relational_fallback": False,
        }

    def _static_measurement_plausible(self, position):
        return bool(
            np.all(position >= STATIC_WORKSPACE_MIN)
            and np.all(position <= STATIC_WORKSPACE_MAX)
        )

    def _gripper_aperture(self):
        qpos = np.asarray(
            self._current_obs["robot_state"]["gripper"]["qpos"],
            dtype=float,
        ).reshape(-1, 2)[0]
        return float(abs(qpos[0] - qpos[1]))

    def _apply_occlusion_tracking(self, positions, observations):
        """Propagate an occluded grasped bowl using robot proprioception.

        The propagation is deliberately short-range and gated by both a
        closed gripper and proximity to the last visual target. It uses no
        simulator object state.
        """
        eef = eef_position_from_obs(self._current_obs)
        aperture = self._gripper_aperture()

        if aperture >= GRIPPER_RELEASE_APERTURE:
            self._target_attached = False

        target = self.last_positions.get("target_bowl")
        if (
            not self._target_attached
            and target is not None
            and aperture <= GRIPPER_ATTACH_APERTURE
            and np.linalg.norm(eef - target) <= TARGET_ATTACH_DISTANCE
        ):
            self._target_attached = True

        target_observation = observations.get("target_bowl")
        target_was_measured = bool(
            target_observation and not target_observation["stale"]
        )
        if (
            self._target_attached
            and not target_was_measured
            and target is not None
            and self._last_eef is not None
        ):
            propagated = target + (eef - self._last_eef)
            self.last_positions["target_bowl"] = propagated.copy()
            positions["target_bowl"] = propagated
            target_observation["propagated"] = True

        self._last_eef = eef.copy()
        return positions, observations

    def _backproject(self, detection, depth, intrinsics, camera_position, camera_rotation, correction):
        mask = detection["mask"]
        points = backproject_mask(
            mask,
            depth,
            intrinsics,
            camera_position,
            camera_rotation,
            y_sign=1.0,
        )
        surface = estimate(points)
        return None if surface is None else surface + correction

    def _assign_bowl_slots(self, bowl_measurements, ramekin_position):
        slots = ("target_bowl", "distractor_bowl")
        if not all(slot in self.last_positions for slot in slots):
            ordered = sorted(
                bowl_measurements,
                key=lambda item: np.linalg.norm(
                    (item[1] - ramekin_position)[:2]
                ),
            )
            return dict(zip(slots, ordered, strict=True))

        best_cost = float("inf")
        best = None
        for assignment in itertools.permutations(bowl_measurements):
            cost = sum(
                np.linalg.norm(item[1] - self.last_positions[slot])
                for slot, item in zip(slots, assignment, strict=True)
            )
            if cost < best_cost:
                best_cost = cost
                best = assignment
        return dict(zip(slots, best, strict=True))

    def _assign_single_bowl(self, measurement):
        """Associate one visible bowl with the two persistent bowl tracks."""
        position = measurement[1]
        distances = {
            key: float(np.linalg.norm(position - self.last_positions[key]))
            for key in ("target_bowl", "distractor_bowl")
            if key in self.last_positions
        }
        if not distances:
            return None

        if (
            distances.get("distractor_bowl", float("inf"))
            <= DISTRACTOR_ASSOCIATION_DISTANCE
        ):
            return "distractor_bowl"
        if self._target_attached:
            return "target_bowl"
        if (
            distances.get("target_bowl", float("inf"))
            <= TARGET_ASSOCIATION_DISTANCE
        ):
            return "target_bowl"
        return min(distances, key=distances.get)

    def _bowl_measurement_innovation(self, slot, position):
        """Return distance from a bowl track's proprioceptive prediction."""
        reference = self.last_positions.get(slot)
        if reference is None:
            return 0.0
        reference = reference.copy()
        if (
            slot == "target_bowl"
            and self._target_attached
            and self._last_eef is not None
        ):
            current_eef = eef_position_from_obs(self._current_obs)
            reference += current_eef - self._last_eef
        return float(np.linalg.norm(position - reference))

    def _accept_bowl_measurement(self, slot, position):
        innovation = self._bowl_measurement_innovation(slot, position)
        limit = (
            TARGET_MEASUREMENT_INNOVATION_LIMIT
            if slot == "target_bowl"
            else DISTRACTOR_MEASUREMENT_INNOVATION_LIMIT
        )
        return innovation <= limit, innovation

    def _observe_objects(self):
        should_infer = (
            not self.last_positions
            or self._frame_index % self.perception_interval == 0
        )
        if not should_infer:
            positions, observations = self._held_observations()
            return self._apply_occlusion_tracking(positions, observations)

        image = np.asarray(self._current_obs["pixels"]["image"])[0]
        prediction = self.perception.predict(image)
        self._last_timing = prediction["timing_ms"]

        regular_counts = {
            label: sum(
                item["label"] == label
                for item in prediction["detections"]
            )
            for label in ("black bowl", "white plate", "white ramekin")
        }
        initial_incomplete = (
            not self.last_positions
            and (
                regular_counts["black bowl"] < 2
                or regular_counts["white plate"] < 1
                or regular_counts["white ramekin"] < 1
            )
        )
        if initial_incomplete:
            fallback = self.perception.predict(
                image,
                class_thresholds=INITIALIZATION_CLASS_THRESHOLDS,
                box_threshold=INITIALIZATION_BOX_THRESHOLD,
                text_threshold=INITIALIZATION_TEXT_THRESHOLD,
            )
            # Keep the better result independently for each class. A
            # permissive post-processing pass must never erase a valid
            # detection from the normal pass.
            merged = []
            for label in ("black bowl", "white plate", "white ramekin"):
                regular_items = [
                    item
                    for item in prediction["detections"]
                    if item["label"] == label
                ]
                fallback_items = [
                    item
                    for item in fallback["detections"]
                    if item["label"] == label
                ]
                merged.extend(
                    fallback_items
                    if len(fallback_items) > len(regular_items)
                    else regular_items
                )
            regular_timing = self._last_timing
            prediction = dict(fallback)
            prediction["detections"] = merged
            self._initialization_fallback_used = True
            self._last_timing = {
                key: prediction["timing_ms"][key]
                + regular_timing[key]
                for key in ("detector", "sam", "total")
            }
        grouped = {"black bowl": [], "white plate": [], "white ramekin": []}
        for detection in prediction["detections"]:
            if detection["label"] in grouped:
                grouped[detection["label"]].append(detection)
        self._last_detection_counts = {
            key: len(value) for key, value in grouped.items()
        }

        eye_bowl_measurements = []
        eye_raw_detections = []
        use_eye_camera = (
            (not self.last_positions and len(grouped["black bowl"]) < 2)
            or (
                bool(self.last_positions)
                and self._target_attached
                and len(grouped["black bowl"]) < 2
            )
        )
        if use_eye_camera:
            eye_image = np.asarray(
                self._current_obs["pixels"]["image2"]
            )[0]
            eye_prediction = self.perception.predict(
                eye_image,
                class_thresholds=INITIALIZATION_CLASS_THRESHOLDS,
                box_threshold=INITIALIZATION_BOX_THRESHOLD,
                text_threshold=INITIALIZATION_TEXT_THRESHOLD,
            )
            eye_counts = {
                label: sum(
                    item["label"] == label
                    for item in eye_prediction["detections"]
                )
                for label in (
                    "black bowl",
                    "white plate",
                    "white ramekin",
                )
            }
            if not self.last_positions:
                self._initialization_eye_used = True
                self._initialization_eye_counts = eye_counts
            eye_raw_detections = eye_prediction.get("raw_detections", [])
            self._last_timing = {
                key: self._last_timing[key]
                + eye_prediction["timing_ms"][key]
                for key in ("detector", "sam", "total")
            }

            _, eye_raw_depth = self.sim.render(
                camera_name=EYE_CAMERA,
                width=WIDTH,
                height=HEIGHT,
                depth=True,
            )
            eye_depth, _, _ = depth_to_meters(
                self.sim, eye_raw_depth
            )
            eye_intrinsics, eye_position, eye_rotation = camera_calibration(
                self.sim, EYE_CAMERA, WIDTH, HEIGHT
            )
            for detection in eye_prediction["detections"]:
                if detection["label"] != "black bowl":
                    continue
                if (
                    self._target_attached
                    and detection["score"] < EYE_HELD_MIN_DETECTION_SCORE
                ):
                    continue
                eye_correction = (
                    EYE_HELD_BOWL_POSITION_CORRECTION
                    if self._target_attached
                    else EYE_BOWL_POSITION_CORRECTION
                )
                position = self._backproject(
                    detection,
                    eye_depth,
                    eye_intrinsics,
                    eye_position,
                    eye_rotation,
                    eye_correction,
                )
                if position is None:
                    continue
                detection = dict(detection)
                detection["camera"] = EYE_CAMERA
                eye_bowl_measurements.append((detection, position))

        _, raw_depth = self.sim.render(
            camera_name=CAMERA,
            width=WIDTH,
            height=HEIGHT,
            depth=True,
        )
        depth, _, _ = depth_to_meters(self.sim, raw_depth)
        intrinsics, camera_position, camera_rotation = camera_calibration(
            self.sim, CAMERA, WIDTH, HEIGHT
        )

        positions, observations = self._held_observations(
            perception_frame=True
        )
        measured = {}
        for label, key in (
            ("white plate", "plate"),
            ("white ramekin", "ramekin"),
        ):
            if not grouped[label]:
                continue
            detection = grouped[label][0]
            position = self._backproject(
                detection,
                depth,
                intrinsics,
                camera_position,
                camera_rotation,
                POSITION_CORRECTIONS[key],
            )
            if position is None:
                continue
            if not self._static_measurement_plausible(position):
                observation = self._fresh_observation(detection)
                observation.update(
                    {
                        "visible": False,
                        "stale": True,
                        "workspace_rejected": True,
                    }
                )
                observations[key] = observation
                continue
            measured[key] = position
            observation = self._fresh_observation(detection)
            observation["anchored"] = True
            observations[key] = observation

        bowl_measurements = []
        for detection in grouped["black bowl"]:
            detection = dict(detection)
            detection["camera"] = CAMERA
            position = self._backproject(
                detection,
                depth,
                intrinsics,
                camera_position,
                camera_rotation,
                POSITION_CORRECTIONS["target_bowl"],
            )
            if position is None:
                continue
            bowl_measurements.append((detection, position))

        for detection, position in eye_bowl_measurements:
            duplicate = any(
                np.linalg.norm(position - existing_position)
                <= MULTIVIEW_DEDUPLICATION_DISTANCE
                for _, existing_position in bowl_measurements
            )
            if not duplicate:
                bowl_measurements.append((detection, position))

        if not self.last_positions:
            self._last_detection_counts["black bowl"] = len(
                bowl_measurements
            )

        bowl_slots = {}
        initial_eye_only = [
            item
            for item in bowl_measurements
            if item[0].get("camera") == EYE_CAMERA
        ]
        initial_agent = [
            item
            for item in bowl_measurements
            if item[0].get("camera") == CAMERA
        ]
        if (
            not self.last_positions
            and len(initial_eye_only) == 1
            and len(initial_agent) >= 1
        ):
            # In the reset pose, the wrist camera's unique non-duplicate
            # bowl is the task target directly below the gripper; agentview's
            # single high-confidence bowl is the static distractor.
            bowl_slots = {
                "target_bowl": initial_eye_only[0],
                "distractor_bowl": initial_agent[0],
            }
        elif len(bowl_measurements) >= 2:
            ramekin_position = measured.get(
                "ramekin", self.last_positions.get("ramekin")
            )
            if ramekin_position is None and not self.last_positions:
                # Reset-pose fallback: the instructed target is the bowl
                # directly in front of / nearest to the end effector.
                ramekin_position = eef_position_from_obs(self._current_obs)
            if ramekin_position is not None:
                bowl_slots = self._assign_bowl_slots(
                    bowl_measurements[:2], ramekin_position
                )
        elif len(bowl_measurements) == 1:
            slot = self._assign_single_bowl(bowl_measurements[0])
            if slot is not None:
                bowl_slots[slot] = bowl_measurements[0]

        for slot, (detection, position) in bowl_slots.items():
            accepted, innovation = self._accept_bowl_measurement(
                slot, position
            )
            if (
                slot == "target_bowl"
                and self._target_attached
                and detection.get("camera") == EYE_CAMERA
                and innovation > EYE_HELD_TARGET_INNOVATION_LIMIT
            ):
                accepted = False
            if not accepted:
                observations[slot].update(
                    {
                        "pixel_count": int(detection["mask"].sum()),
                        "detection_score": float(detection["score"]),
                        "sam_predicted_iou": float(
                            detection["sam_predicted_iou"]
                        ),
                        "measurement_rejected": True,
                        "measurement_innovation": innovation,
                    }
                )
                continue
            positions[slot] = position
            self.last_positions[slot] = position.copy()
            self._last_seen_frame[slot] = self._frame_index
            observations[slot] = self._fresh_observation(detection)
            observations[slot]["measurement_innovation"] = innovation

        if (
            "ramekin" not in self.last_positions
            and "ramekin" not in measured
            and "target_bowl" in positions
        ):
            # Task-language relation fallback for an occluded / confused
            # ramekin detection. This is a calibrated object-frame relation,
            # not simulator state.
            inferred = (
                positions["target_bowl"] + RAMEKIN_FROM_TARGET_OFFSET
            )
            positions["ramekin"] = inferred
            self.last_positions["ramekin"] = inferred.copy()
            self._ramekin_relational_anchor = True
            observations["ramekin"].update(
                {
                    "anchored": True,
                    "relational_fallback": True,
                    "last_seen_frame": self._frame_index,
                }
            )

        for key in ("plate", "ramekin"):
            if key not in measured:
                continue
            position = measured[key]
            if key in self.last_positions:
                # Both landmarks are fixed in this LIBERO task. Learned masks
                # can merge the plate with a carried bowl or gripper during
                # recovery, producing a false 3--5 cm relocalization. Retain
                # the accurate reset-time RGB-D anchor and use later frames
                # only as visibility evidence.
                position = self.last_positions[key].copy()
                observations[key]["anchor_frozen"] = True
            else:
                self.last_positions[key] = position.copy()
                observations[key]["anchor_frozen"] = False
            positions[key] = position
            self._last_seen_frame[key] = self._frame_index
            observations[key]["last_seen_frame"] = self._frame_index

        if not set(OBJECT_KEYS).issubset(positions):
            missing = sorted(set(OBJECT_KEYS) - set(positions))
            raw = prediction.get("raw_detections", [])
            candidates = [
                {
                    "label": item["label"],
                    "score": round(float(item["score"]), 4),
                }
                for item in raw
            ]
            eye_candidates = [
                {
                    "label": item["label"],
                    "score": round(float(item["score"]), 4),
                }
                for item in eye_raw_detections
            ]
            raise RuntimeError(
                "Initial learned perception cannot initialize objects: "
                f"missing={missing}, counts={self._last_detection_counts}, "
                f"fallback_raw={candidates}, eye_counts="
                f"{self._initialization_eye_counts}, "
                f"eye_raw={eye_candidates}"
            )

        return self._apply_occlusion_tracking(positions, observations)

    def update(self, obs):
        self._current_obs = obs
        was_target_attached = self._target_attached
        scene = super().update(obs)

        # The oracle state recognizes a placement attempt while the lifted
        # bowl passes directly over the plate. Learned RGB-D can miss that
        # brief event because inference runs every few control steps and the
        # bowl is often occluded by the gripper. Releasing a previously held
        # target inside a generous plate workspace is equivalent evidence
        # that the task has entered placement verification.
        released_near_plate = (
            was_target_attached
            and not self._target_attached
            and self.ever_held
            and scene["bowl_plate_xy_distance"]
            < PLACEMENT_RELEASE_XY_LIMIT
            and -0.02 < scene["bowl_plate_z_distance"] < 0.12
        )
        self._placement_attempted = (
            self._placement_attempted
            or self.was_over_plate
            or released_near_plate
        )
        if self._placement_attempted:
            self.was_over_plate = True

        # LIBERO terminates successful episodes immediately. Learned masks
        # update less frequently than oracle masks, so only two settled
        # samples may remain before termination. Keep the calibrated 3.2 cm
        # geometric threshold, but use a two-frame temporal confirmation.
        learned_on_plate = (
            scene["on_plate_raw"]
            and scene["on_plate_count"]
            >= LEARNED_ON_PLATE_CONFIRMATION_STEPS
        )

        # After a previously held target is released near the plate, allow a
        # fixed verification window for the success predicate. Do not require
        # noisy RGB-D distance and height estimates to remain simultaneously
        # inside a failure region for consecutive frames: that caused the
        # failure counter to reset for ~100 steps in a valid failed placement.
        if learned_on_plate:
            self._placement_verification_count = 0
        elif (
            self._placement_attempted
            and not self._target_attached
            and not scene["held"]
        ):
            self._placement_verification_count += 1

        learned_placement_failed_raw = (
            self._placement_verification_count > 0
            and not learned_on_plate
        )
        learned_placement_failed = (
            self._placement_verification_count
            >= PLACEMENT_VERIFICATION_TIMEOUT_STEPS
        )
        self.placement_failure_count = self._placement_verification_count
        scene["placement_failed_raw"] = learned_placement_failed_raw
        scene["placement_failure_count"] = self.placement_failure_count
        scene["placement_verification_count"] = (
            self._placement_verification_count
        )
        scene["placement_failed"] = learned_placement_failed

        if learned_on_plate:
            scene["on_plate"] = True
            scene["placement_failed"] = False
            scene["stage"] = "complete"
        elif learned_placement_failed:
            scene["stage"] = "placement_failed"
        elif self._placement_attempted and not scene["held"]:
            # Task progress is monotonic: loss of a released target must not
            # send verification back to approach/grasp.
            scene["stage"] = "verify_place"

        # Once the target has settled in the plate envelope, it should no
        # longer inherit end-effector motion even if the gripper remains
        # mechanically closed after placing the object.
        if scene["on_plate_raw"] and not scene["held"]:
            self._target_attached = False

        scene["source"] = "agentview_rgbd_grounding_dino_sam"
        scene["perception_interval"] = self.perception_interval
        scene["perception_frame_index"] = self._frame_index
        scene["last_inference_timing_ms"] = self._last_timing
        scene["last_detection_counts"] = self._last_detection_counts
        scene["target_attached"] = self._target_attached
        scene["placement_attempted"] = self._placement_attempted
        scene["initialization_fallback_used"] = (
            self._initialization_fallback_used
        )
        scene["initialization_eye_used"] = self._initialization_eye_used
        scene["initialization_eye_counts"] = self._initialization_eye_counts
        scene["ramekin_relational_anchor"] = (
            self._ramekin_relational_anchor
        )
        self._frame_index += 1
        return scene

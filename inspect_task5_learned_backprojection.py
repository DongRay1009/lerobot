"""Evaluate learned task-5 masks through RGB-D 3D backprojection."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from inspect_task5_backprojection import (
    backproject_mask,
    camera_calibration,
    estimate,
)
from inspect_task5_masks import geom_ids_for_body_tree
from inspect_task5_vision import depth_to_meters, robosuite_segmentation
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from scene_memory import TASK5_BODIES, body_id, body_position, find_sim
from vision_scene_memory import POSITION_CORRECTIONS


CAMERA = "agentview"
WIDTH = 360
HEIGHT = 360
IMAGE_PATH = Path("vision_debug/obs_image.png")
MASK_DIR = Path("vision_debug/learned_masks")
MASK_REPORT = MASK_DIR / "mask_report.json"
OUT = Path("vision_debug/learned_backprojection_report.json")


def error(left, right):
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return {
        "delta_xyz": delta.round(6).tolist(),
        "error_3d_m": float(np.linalg.norm(delta)),
        "error_xy_m": float(np.linalg.norm(delta[:2])),
        "error_z_m": float(abs(delta[2])),
    }


def mask_iou(left, right):
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Saved RGB image not found: {IMAGE_PATH}")
    if not MASK_REPORT.is_file():
        raise FileNotFoundError(f"Learned mask report not found: {MASK_REPORT}")

    mask_report = json.loads(MASK_REPORT.read_text())
    learned_entries = {
        item["matched_object"]: item
        for item in mask_report["masks"]
        if item.get("matched_object")
    }
    required = {"target_bowl", "distractor_bowl", "ramekin", "plate"}
    if not required.issubset(learned_entries):
        missing = sorted(required - set(learned_entries))
        raise RuntimeError(f"Learned masks missing evaluated objects: {missing}")

    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    container = all_envs["libero_spatial"]
    env = next(iter(container.values())) if isinstance(container, dict) else container[0]

    report = {"camera": CAMERA, "objects": {}}
    try:
        env.reset()
        sim = find_sim(env)
        rendered_rgb, raw_depth = sim.render(
            camera_name=CAMERA,
            width=WIDTH,
            height=HEIGHT,
            depth=True,
        )
        saved_rgb = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
        rgb_mae = float(
            np.mean(
                np.abs(
                    rendered_rgb.astype(np.float32)
                    - saved_rgb.astype(np.float32)
                )
            )
        )
        report["fresh_vs_saved_rgb_mae"] = rgb_mae
        report["fresh_vs_saved_rgb_mae_fraction"] = rgb_mae / 255.0
        print(f"fresh-vs-saved RGB MAE: {rgb_mae:.4f}")
        if rgb_mae > 5.0:
            print("WARNING: fresh reset differs from the saved mask image")

        depth, _, _ = depth_to_meters(sim, raw_depth)
        segmentation = robosuite_segmentation(sim, CAMERA)
        intrinsics, camera_position, camera_rotation = camera_calibration(
            sim, CAMERA, WIDTH, HEIGHT
        )
        geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)

        for key in sorted(required):
            root_id = body_id(sim.model, TASK5_BODIES[key])
            geom_ids = geom_ids_for_body_tree(sim.model, root_id)
            oracle_mask = (
                (segmentation[:, :, 0] == geom_objtype)
                & np.isin(segmentation[:, :, 1], geom_ids)
            )
            learned_path = MASK_DIR / learned_entries[key]["mask_file"]
            learned_mask = np.asarray(Image.open(learned_path).convert("L")) > 0

            learned_points = backproject_mask(
                learned_mask,
                depth,
                intrinsics,
                camera_position,
                camera_rotation,
                y_sign=1.0,
            )
            oracle_points = backproject_mask(
                oracle_mask,
                depth,
                intrinsics,
                camera_position,
                camera_rotation,
                y_sign=1.0,
            )
            learned_position = estimate(learned_points) + POSITION_CORRECTIONS[key]
            oracle_position = estimate(oracle_points) + POSITION_CORRECTIONS[key]
            truth = body_position(sim, root_id)

            object_report = {
                "learned_mask_pixels": int(learned_mask.sum()),
                "oracle_mask_pixels": int(oracle_mask.sum()),
                "saved_mask_iou": learned_entries[key]["oracle_iou"],
                "fresh_mask_iou": mask_iou(learned_mask, oracle_mask),
                "learned_position_xyz": learned_position.round(6).tolist(),
                "oracle_position_xyz": oracle_position.round(6).tolist(),
                "truth_position_xyz": truth.round(6).tolist(),
                "learned_vs_truth": error(learned_position, truth),
                "oracle_vs_truth": error(oracle_position, truth),
                "learned_vs_oracle": error(learned_position, oracle_position),
            }
            report["objects"][key] = object_report
            print(
                f"{key}: saved/fresh_IoU="
                f"{object_report['saved_mask_iou']:.3f}/"
                f"{object_report['fresh_mask_iou']:.3f} "
                f"learned_truth={object_report['learned_vs_truth']['error_3d_m'] * 100:.2f}cm "
                f"oracle_truth={object_report['oracle_vs_truth']['error_3d_m'] * 100:.2f}cm "
                f"learned_oracle={object_report['learned_vs_oracle']['error_3d_m'] * 100:.2f}cm"
            )

        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print("report:", OUT.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()

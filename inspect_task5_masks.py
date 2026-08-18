"""Map MuJoCo segmentation labels to task-5 object masks.

This is a perception debugging tool. MuJoCo segmentation is used only as
ground truth for validating RGB-D geometry and future learned perception.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from inspect_task5_vision import depth_to_meters, robosuite_segmentation
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from scene_memory import TASK5_BODIES, body_id, find_sim


OUT = Path("vision_debug/masks")
OBJECT_KEYS = ("target_bowl", "distractor_bowl", "ramekin", "plate")
COLORS = {
    "target_bowl": np.array([255, 70, 70], dtype=np.float32),
    "distractor_bowl": np.array([255, 190, 60], dtype=np.float32),
    "ramekin": np.array([70, 220, 120], dtype=np.float32),
    "plate": np.array([70, 150, 255], dtype=np.float32),
}


def descendant_body_ids(model, root_id):
    """Return the root body and every descendant body in model order."""
    result = {int(root_id)}
    for index, parent in enumerate(np.asarray(model.body_parentid, dtype=int)):
        if int(parent) in result:
            result.add(index)
    return np.asarray(sorted(result), dtype=int)


def geom_ids_for_body_tree(model, root_id):
    body_ids = descendant_body_ids(model, root_id)
    geom_body_ids = np.asarray(model.geom_bodyid, dtype=int)
    return np.flatnonzero(np.isin(geom_body_ids, body_ids))


def mask_summary(mask, metric_depth):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return {
            "visible": False,
            "pixel_count": 0,
            "centroid_uv": None,
            "bbox_xyxy": None,
            "median_depth_m": None,
        }

    depths = np.asarray(metric_depth[mask], dtype=float)
    depths = depths[np.isfinite(depths) & (depths > 0)]
    return {
        "visible": True,
        "pixel_count": int(len(rows)),
        "centroid_uv": [float(cols.mean()), float(rows.mean())],
        "bbox_xyxy": [
            int(cols.min()),
            int(rows.min()),
            int(cols.max()),
            int(rows.max()),
        ],
        "median_depth_m": float(np.median(depths)) if len(depths) else None,
    }


def save_mask(path, mask):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(path, rgb, masks):
    overlay = np.asarray(rgb, dtype=np.float32).copy()
    for key, mask in masks.items():
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * COLORS[key]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    container = all_envs["libero_spatial"]
    env = next(iter(container.values())) if isinstance(container, dict) else container[0]

    report = {}
    try:
        env.reset()
        sim = find_sim(env)
        geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)

        object_geom_ids = {}
        for key in OBJECT_KEYS:
            root_id = body_id(sim.model, TASK5_BODIES[key])
            geom_ids = geom_ids_for_body_tree(sim.model, root_id)
            object_geom_ids[key] = geom_ids
            names = [sim.model.geom_id2name(int(index)) for index in geom_ids]
            print(
                f"{key}: body={TASK5_BODIES[key]!r} root_id={root_id} "
                f"geom_ids={geom_ids.tolist()} geom_names={names}"
            )

        for camera in ("agentview", "robot0_eye_in_hand"):
            rgb, raw_depth = sim.render(
                camera_name=camera,
                width=360,
                height=360,
                depth=True,
            )
            metric_depth, _, _ = depth_to_meters(sim, raw_depth)
            segmentation = robosuite_segmentation(sim, camera)

            masks = {}
            camera_report = {}
            for key, geom_ids in object_geom_ids.items():
                mask = (
                    (segmentation[:, :, 0] == geom_objtype)
                    & np.isin(segmentation[:, :, 1], geom_ids)
                )
                masks[key] = mask
                summary = mask_summary(mask, metric_depth)
                camera_report[key] = summary
                save_mask(OUT / f"{camera}_{key}.png", mask)
                print(f"{camera} {key}: {summary}")

            save_overlay(OUT / f"{camera}_overlay.png", rgb, masks)
            report[camera] = camera_report

        with (OUT / "mask_report.json").open("w") as file:
            json.dump(report, file, indent=2)
        print(f"saved object masks under: {OUT.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

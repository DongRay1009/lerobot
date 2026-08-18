"""Validate RGB-D backprojection against task-5 MuJoCo body positions."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from inspect_task5_masks import OBJECT_KEYS, geom_ids_for_body_tree
from inspect_task5_vision import depth_to_meters, robosuite_segmentation
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from scene_memory import TASK5_BODIES, body_id, body_position, find_sim


OUT = Path("vision_debug/backprojection_report.json")
WIDTH = 360
HEIGHT = 360


def camera_calibration(sim, camera, width=WIDTH, height=HEIGHT):
    camera_id = sim.model.camera_name2id(camera)
    fovy_deg = float(sim.model.cam_fovy[camera_id])
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    intrinsics = {
        "fx": float(focal),
        "fy": float(focal),
        "cx": float((width - 1) / 2.0),
        "cy": float((height - 1) / 2.0),
        "fovy_deg": fovy_deg,
    }
    position = np.asarray(sim.data.cam_xpos[camera_id], dtype=float).copy()
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy()
    return intrinsics, position, rotation


def backproject_mask(mask, depth, intrinsics, camera_position, camera_rotation, y_sign):
    rows, cols = np.nonzero(mask)
    z = np.asarray(depth[rows, cols], dtype=float)
    valid = np.isfinite(z) & (z > 0)
    rows = rows[valid].astype(float)
    cols = cols[valid].astype(float)
    z = z[valid]
    if len(z) == 0:
        return np.empty((0, 3), dtype=float)

    x_camera = (cols - intrinsics["cx"]) * z / intrinsics["fx"]
    y_camera = y_sign * (rows - intrinsics["cy"]) * z / intrinsics["fy"]
    # MuJoCo cameras look along local -Z. y_sign=+1 corresponds to the
    # bottom-left framebuffer convention; -1 corresponds to top-left images.
    points_camera = np.column_stack((x_camera, y_camera, -z))
    return points_camera @ camera_rotation.T + camera_position


def estimate(points):
    if len(points) == 0:
        return None
    # Coordinate-wise median is robust to a few boundary/background pixels.
    return np.median(points, axis=0)


def error_summary(estimate_position, truth):
    if estimate_position is None:
        return None
    delta = np.asarray(estimate_position) - np.asarray(truth)
    return {
        "estimate_xyz": np.asarray(estimate_position).round(6).tolist(),
        "truth_xyz": np.asarray(truth).round(6).tolist(),
        "delta_xyz": delta.round(6).tolist(),
        "error_3d_m": float(np.linalg.norm(delta)),
        "error_xy_m": float(np.linalg.norm(delta[:2])),
        "error_z_m": float(abs(delta[2])),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    container = all_envs["libero_spatial"]
    env = next(iter(container.values())) if isinstance(container, dict) else container[0]

    report = {}
    try:
        env.reset()
        sim = find_sim(env)
        geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)

        roots = {
            key: body_id(sim.model, TASK5_BODIES[key])
            for key in OBJECT_KEYS
        }
        geom_ids = {
            key: geom_ids_for_body_tree(sim.model, root_id)
            for key, root_id in roots.items()
        }

        for camera in ("agentview", "robot0_eye_in_hand"):
            _, raw_depth = sim.render(
                camera_name=camera,
                width=WIDTH,
                height=HEIGHT,
                depth=True,
            )
            depth, _, _ = depth_to_meters(sim, raw_depth)
            segmentation = robosuite_segmentation(sim, camera)
            intrinsics, camera_position, camera_rotation = camera_calibration(sim, camera)

            print(f"\n{camera}")
            print("intrinsics:", {key: round(value, 4) for key, value in intrinsics.items()})
            print("camera_position:", camera_position.round(6).tolist())

            camera_report = {
                "intrinsics": intrinsics,
                "camera_position": camera_position.tolist(),
                "objects": {},
            }
            for key in OBJECT_KEYS:
                mask = (
                    (segmentation[:, :, 0] == geom_objtype)
                    & np.isin(segmentation[:, :, 1], geom_ids[key])
                )
                truth = body_position(sim, roots[key])

                conventions = {}
                for convention, y_sign in (("bottom_left", 1.0), ("top_left", -1.0)):
                    points = backproject_mask(
                        mask,
                        depth,
                        intrinsics,
                        camera_position,
                        camera_rotation,
                        y_sign,
                    )
                    conventions[convention] = error_summary(estimate(points), truth)

                best = min(
                    conventions,
                    key=lambda name: conventions[name]["error_3d_m"],
                )
                camera_report["objects"][key] = {
                    "pixel_count": int(mask.sum()),
                    "best_convention": best,
                    "conventions": conventions,
                }
                best_result = conventions[best]
                print(
                    f"{key}: pixels={int(mask.sum())} best={best} "
                    f"estimate={best_result['estimate_xyz']} "
                    f"truth={best_result['truth_xyz']} "
                    f"error={best_result['error_3d_m'] * 100:.2f}cm "
                    f"xy={best_result['error_xy_m'] * 100:.2f}cm "
                    f"z={best_result['error_z_m'] * 100:.2f}cm"
                )

            report[camera] = camera_report

        with OUT.open("w") as file:
            json.dump(report, file, indent=2)
        print(f"\nsaved report: {OUT.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

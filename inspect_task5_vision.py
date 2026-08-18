"""Inspect RGB, depth, segmentation, and camera metadata for task 5."""

from __future__ import annotations

import inspect
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from scene_memory import find_sim


OUT = Path("vision_debug")


def model_names(model, kind, count):
    converter = getattr(model, f"{kind}_id2name", None)
    names = []
    for index in range(count):
        name = None
        if callable(converter):
            try:
                name = converter(index)
            except Exception:
                pass
        names.append(name)
    return names


def save_rgb(name, array):
    image = np.asarray(array)
    if image.ndim == 4:
        image = image[0]
    image = np.ascontiguousarray(image)
    Image.fromarray(image.astype(np.uint8)).save(OUT / name)


def native_model_and_data(sim):
    """Unwrap robosuite's compatibility objects for the native MuJoCo API."""
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def depth_to_meters(sim, depth):
    """Convert MuJoCo's normalized OpenGL depth buffer to metric depth."""
    model, _ = native_model_and_data(sim)
    near = float(model.vis.map.znear * model.stat.extent)
    far = float(model.vis.map.zfar * model.stat.extent)
    depth = np.asarray(depth, dtype=np.float32)
    metric = near / (1.0 - depth * (1.0 - near / far))
    return metric, near, far


def robosuite_segmentation(sim, camera, width=360, height=360):
    """Use robosuite's EGL context with NumPy-2-safe ID decoding.

    robosuite 1.4.0 multiplies uint8 channels by 256 and 65536 before
    promoting them, which raises OverflowError with NumPy 2. Convert the
    RGB ID buffer to int32 first while keeping the original render context.
    """
    from robosuite.utils import binding_utils

    camera_id = sim.model.camera_name2id(camera)
    context = sim._render_context_offscreen
    with binding_utils._MjSim_render_lock:
        context.render(
            width=width,
            height=height,
            camera_id=camera_id,
            segmentation=True,
        )
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_ids = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(
            rgb=rgb_ids,
            depth=None,
            viewport=viewport,
            con=context.con,
        )

        rgb_ids = rgb_ids.astype(np.int32)
        packed = (
            rgb_ids[:, :, 0]
            + rgb_ids[:, :, 1] * 256
            + rgb_ids[:, :, 2] * 65536
        )
        packed[packed >= context.scn.ngeom + 1] = 0

        labels = np.full(
            (context.scn.ngeom + 1, 2),
            fill_value=-1,
            dtype=np.int32,
        )
        for index in range(context.scn.ngeom):
            geom = context.scn.geoms[index]
            if geom.segid != -1:
                labels[geom.segid + 1] = (geom.objtype, geom.objid)
        return labels[packed]


def save_segmentation_preview(name, segmentation):
    """Assign a stable color to every segmentation pair for inspection."""
    segmentation = np.asarray(segmentation, dtype=np.int32)
    pairs = segmentation.reshape(-1, segmentation.shape[-1])
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    colors = np.zeros((len(unique), 3), dtype=np.uint8)
    valid = np.all(unique >= 0, axis=1)
    indices = np.arange(len(unique), dtype=np.uint32)
    colors[valid, 0] = (indices[valid] * 53 + 67) % 255
    colors[valid, 1] = (indices[valid] * 97 + 29) % 255
    colors[valid, 2] = (indices[valid] * 193 + 101) % 255
    preview = colors[inverse].reshape(segmentation.shape[:2] + (3,))
    Image.fromarray(preview).save(OUT / name)
    return unique


def main():
    OUT.mkdir(exist_ok=True)

    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    container = all_envs["libero_spatial"]
    env = next(iter(container.values())) if isinstance(container, dict) else container[0]

    try:
        obs, _ = env.reset()
        print("observation pixel keys:", list(obs["pixels"].keys()))
        for key, image in obs["pixels"].items():
            print(f"obs {key}: shape={np.asarray(image).shape} dtype={np.asarray(image).dtype}")
            save_rgb(f"obs_{key}.png", image)

        sim = find_sim(env)
        try:
            print("sim.render signature:", inspect.signature(sim.render))
        except (TypeError, ValueError):
            print("sim.render signature: unavailable")

        camera_names = model_names(sim.model, "camera", sim.model.ncam)
        print("camera names:", camera_names)

        for camera in ("agentview", "robot0_eye_in_hand"):
            try:
                rendered = sim.render(
                    camera_name=camera,
                    width=360,
                    height=360,
                    depth=True,
                )
                if isinstance(rendered, tuple):
                    rgb, depth = rendered[:2]
                    metric_depth, near, far = depth_to_meters(sim, depth)
                    print(
                        f"{camera} depth render: rgb={np.asarray(rgb).shape} "
                        f"depth={np.asarray(depth).shape} "
                        f"raw_range=({np.min(depth):.5f}, {np.max(depth):.5f}) "
                        f"metric_range=({np.min(metric_depth):.4f}, "
                        f"{np.max(metric_depth):.4f})m clip=({near:.4f}, {far:.1f})m"
                    )
                    save_rgb(f"render_{camera}.png", rgb)
                    np.save(OUT / f"depth_raw_{camera}.npy", depth)
                    np.save(OUT / f"depth_meters_{camera}.npy", metric_depth)
                else:
                    print(f"{camera} depth render returned: {np.asarray(rendered).shape}")
            except Exception as exc:
                print(f"{camera} depth render failed: {type(exc).__name__}: {exc}")

            try:
                segmentation = robosuite_segmentation(sim, camera)
                print(
                    f"{camera} segmentation: shape={segmentation.shape} "
                    f"dtype={segmentation.dtype}"
                )
                if segmentation.ndim >= 3 and segmentation.shape[-1] >= 2:
                    pairs = np.unique(segmentation.reshape(-1, segmentation.shape[-1]), axis=0)
                    print(f"{camera} unique segmentation pairs: {pairs[:30].tolist()}")
                np.save(OUT / f"segmentation_{camera}.npy", segmentation)
                save_segmentation_preview(
                    f"segmentation_{camera}.png",
                    segmentation,
                )
            except Exception as exc:
                print(f"{camera} segmentation failed: {type(exc).__name__}: {exc}")

        print(f"saved inspection artifacts under: {OUT.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

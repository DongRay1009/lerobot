"""Inspect MuJoCo objects for LIBERO Spatial task 5.

This is a read-only discovery utility for phase 2. It creates task 5, walks
through Gym/robosuite wrappers to find the MuJoCo simulator, then prints body
names and poses relevant to the bowl/ramekin/plate task.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv


KEYWORDS = ("bowl", "ramekin", "plate", "gripper", "eef")


def find_sim(root):
    """Breadth-first search through common Gym/vector wrapper attributes."""
    queue = [("root", root)]
    seen = set()

    while queue:
        path, obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        print(f"wrapper: {path}: {type(obj).__module__}.{type(obj).__name__}")

        sim = getattr(obj, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return path + ".sim", sim

        for attr in (
            "env",
            "_env",
            "venv",
            "_venv",
            "unwrapped",
            "wrapped_env",
            "_wrapped_env",
            "libero_env",
            "_libero_env",
        ):
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None and child is not obj:
                queue.append((f"{path}.{attr}", child))

        children = getattr(obj, "envs", None)
        if children:
            if isinstance(children, dict):
                for key, child in list(children.items()):
                    queue.append((f"{path}.envs[{key!r}]", child))
            else:
                for index, child in enumerate(list(children)):
                    queue.append((f"{path}.envs[{index}]", child))

        # Library wrappers occasionally use version-specific private names.
        # Discover only attributes that look like environment/simulator links;
        # avoid traversing arbitrary model/data objects.
        try:
            attributes = vars(obj)
        except TypeError:
            attributes = {}
        for attr, child in attributes.items():
            lowered = attr.lower()
            if child is None or child is obj:
                continue
            if "env" in lowered or "sim" in lowered:
                queue.append((f"{path}.{attr}", child))

    raise RuntimeError("Could not find an object exposing .sim.model and .sim.data")


def body_names(model):
    names = []
    for body_id in range(model.nbody):
        name = None

        id_to_name = getattr(model, "body_id2name", None)
        if callable(id_to_name):
            try:
                name = id_to_name(body_id)
            except Exception:
                pass

        if not name:
            all_names = getattr(model, "body_names", None)
            if all_names is not None and body_id < len(all_names):
                name = all_names[body_id]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

        if not name:
            native_model = getattr(model, "_model", model)
            try:
                name = mujoco.mj_id2name(
                    native_model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
            except (TypeError, AttributeError):
                pass

        if name:
            names.append((body_id, name))
    return names


def body_pose(data, body_id):
    positions = getattr(data, "body_xpos", None)
    quaternions = getattr(data, "body_xquat", None)
    if positions is None:
        positions = getattr(data, "xpos")
    if quaternions is None:
        quaternions = getattr(data, "xquat")
    return np.asarray(positions[body_id]), np.asarray(quaternions[body_id])


def main():
    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    suite_envs = all_envs["libero_spatial"]
    env = next(iter(suite_envs.values())) if isinstance(suite_envs, dict) else suite_envs[0]

    try:
        env.reset()
        sim_path, sim = find_sim(env)
        print(f"\nFound simulator at: {sim_path}")
        print(f"Bodies: {sim.model.nbody}")

        matches = []
        for body_id, name in body_names(sim.model):
            if any(keyword in name.lower() for keyword in KEYWORDS):
                raw_pos, raw_quat = body_pose(sim.data, body_id)
                pos = raw_pos.round(5).tolist()
                quat = raw_quat.round(5).tolist()
                matches.append((body_id, name, pos, quat))

        print("\nRelevant body poses:")
        for body_id, name, pos, quat in matches:
            print(f"id={body_id:3d} name={name!r} pos={pos} quat={quat}")

        if not matches:
            print("No keyword matches. All named bodies follow:")
            for body_id, name in body_names(sim.model):
                print(f"id={body_id:3d} name={name!r}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

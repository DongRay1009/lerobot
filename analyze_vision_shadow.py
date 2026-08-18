"""Compare ground-truth and RGB-D shadow scene memories in JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path

import numpy as np


OBJECTS = ("target_bowl", "distractor_bowl", "ramekin", "plate")
BOOLEAN_FIELDS = ("held", "on_plate", "placement_failed")


def numeric_values(value):
    if isinstance(value, dict):
        if "values" in value:
            return np.asarray(value["values"], dtype=float)
        arrays = [numeric_values(item) for item in value.values()]
        arrays = [item for item in arrays if item.size]
        return np.concatenate(arrays) if arrays else np.asarray([])
    if isinstance(value, list):
        arrays = [numeric_values(item) for item in value]
        arrays = [item for item in arrays if item.size]
        return np.concatenate(arrays) if arrays else np.asarray([])
    if isinstance(value, (int, float, bool)):
        return np.asarray([value], dtype=float)
    return np.asarray([])


def first_true(rows, source, field):
    for row in rows:
        scene = row.get(source)
        if scene and scene.get(field, False):
            return int(row["step"])
    return None


def transitions(rows, source):
    result = []
    previous = None
    for row in rows:
        scene = row.get(source)
        stage = None if scene is None else scene.get("stage")
        if stage is not None and stage != previous:
            result.append(f"{row['step']}:{stage}")
            previous = stage
    return " -> ".join(result)


def percentiles_cm(values):
    values = np.asarray(values, dtype=float) * 100.0
    if len(values) == 0:
        return "no fresh samples"
    return (
        f"median={np.median(values):.2f} "
        f"p95={np.percentile(values, 95):.2f} "
        f"max={np.max(values):.2f}cm"
    )


def observation_key(vision, semantic_key):
    target = vision.get("target_instance", "target_bowl")
    other = "distractor_bowl" if target == "target_bowl" else "target_bowl"
    if semantic_key == "target_bowl":
        return target
    if semantic_key == "distractor_bowl":
        return other
    return semantic_key


def analyze(path):
    records = [json.loads(line) for line in Path(path).open()]
    rows = [
        row
        for row in records
        if row.get("event") == "step"
        and row.get("scene") is not None
        and row.get("vision_scene") is not None
    ]
    success = any(numeric_values(row.get("reward", {})).sum() > 0 for row in rows)
    stage_matches = sum(
        row["scene"].get("stage") == row["vision_scene"].get("stage")
        for row in rows
    )
    target_instances = Counter(
        row["vision_scene"].get("target_instance") for row in rows
    )

    print(f"\n{path}")
    print(
        f"success={success} steps={len(rows)} "
        f"stage_agreement={stage_matches / max(len(rows), 1) * 100:.1f}% "
        f"target_instances={dict(target_instances)}"
    )

    for key in OBJECTS:
        errors_3d = []
        errors_xy = []
        fresh_errors = []
        stale_count = 0
        min_pixels = None
        for row in rows:
            gt = np.asarray(row["scene"]["positions"][key], dtype=float)
            vision = row["vision_scene"]
            estimate = np.asarray(vision["positions"][key], dtype=float)
            delta = estimate - gt
            error = float(np.linalg.norm(delta))
            errors_3d.append(error)
            errors_xy.append(float(np.linalg.norm(delta[:2])))

            obs_key = observation_key(vision, key)
            observation = vision.get("observations", {}).get(obs_key, {})
            stale = bool(observation.get("stale", False))
            pixels = observation.get("pixel_count")
            if pixels is not None:
                min_pixels = pixels if min_pixels is None else min(min_pixels, pixels)
            if stale:
                stale_count += 1
            else:
                fresh_errors.append(error)

        print(
            f"{key}: fresh_3d[{percentiles_cm(fresh_errors)}] "
            f"all_xy_p95={np.percentile(errors_xy, 95) * 100:.2f}cm "
            f"stale={stale_count}/{len(rows)} min_pixels={min_pixels}"
        )

    for field in BOOLEAN_FIELDS:
        matches = sum(
            bool(row["scene"].get(field, False))
            == bool(row["vision_scene"].get(field, False))
            for row in rows
        )
        gt_positive = sum(bool(row["scene"].get(field, False)) for row in rows)
        vision_positive = sum(
            bool(row["vision_scene"].get(field, False)) for row in rows
        )
        print(
            f"{field}: agreement={matches / max(len(rows), 1) * 100:.1f}% "
            f"GT_positive={gt_positive} vision_positive={vision_positive} "
            f"first_GT={first_true(rows, 'scene', field)} "
            f"first_vision={first_true(rows, 'vision_scene', field)}"
        )

    print("GT stages:    ", transitions(rows, "scene"))
    print("vision stages:", transitions(rows, "vision_scene"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    args = parser.parse_args()
    paths = sorted(glob.glob(str(Path(args.trace_dir) / "episode_*.jsonl")))
    if not paths:
        raise SystemExit(f"No episode_*.jsonl files found under {args.trace_dir}")
    for path in paths:
        analyze(path)


if __name__ == "__main__":
    main()

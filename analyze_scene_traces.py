"""Summarize object-relative LIBERO task-5 rollout traces."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def numeric_values(summary):
    if not isinstance(summary, dict):
        return np.asarray([], dtype=float)
    return np.asarray(summary.get("values", []), dtype=float)


def stage_transitions(scenes):
    result = []
    previous = None
    for step, scene in enumerate(scenes):
        stage = scene["stage"]
        if stage != previous:
            result.append(f"{step}:{stage}")
            previous = stage
    return " -> ".join(result)


def longest_true_run(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    steps = [row for row in rows if row.get("event") == "step"]
    scenes = [row.get("scene") for row in steps]

    if not steps or any(scene is None for scene in scenes):
        raise RuntimeError(f"Missing step or scene data in {path}")

    actions = np.asarray([numeric_values(row["action"]) for row in steps])
    close_indices = np.where(actions[:, -1] > 0.8)[0]
    first_close = int(close_indices[0]) if close_indices.size else None

    rewards = [numeric_values(row["reward"]) for row in steps]
    success = any(values.size and float(values.max()) > 0 for values in rewards)

    eef_bowl = np.asarray([scene["eef_bowl_distance"] for scene in scenes])
    lifts = np.asarray([scene["bowl_lift"] for scene in scenes])
    bowl_plate_xy = np.asarray(
        [scene["bowl_plate_xy_distance"] for scene in scenes]
    )
    bowl_plate_delta = np.asarray(
        [
            np.asarray(scene["positions"]["target_bowl"])
            - np.asarray(scene["positions"]["plate"])
            for scene in scenes
        ]
    )
    min_distance_step = int(np.argmin(eef_bowl))
    max_lift_step = int(np.argmax(lifts))
    min_plate_step = int(np.argmin(bowl_plate_xy))
    verified_on_plate = (
        bowl_plate_xy < 0.035
    ) & (
        bowl_plate_delta[:, 2] > -0.005
    ) & (
        bowl_plate_delta[:, 2] < 0.035
    )
    recorded_placement_failures = [
        bool(scene.get("placement_failed", False)) for scene in scenes
    ]

    result = {
        "file": str(path),
        "success": success,
        "steps": len(steps),
        "first_close": first_close,
        "min_eef_bowl_distance": float(eef_bowl[min_distance_step]),
        "min_distance_step": min_distance_step,
        "max_bowl_lift": float(lifts[max_lift_step]),
        "max_lift_step": max_lift_step,
        "min_bowl_plate_xy": float(bowl_plate_xy[min_plate_step]),
        "min_plate_step": min_plate_step,
        "delta_at_min_plate": bowl_plate_delta[min_plate_step],
        "final_bowl_plate_delta": bowl_plate_delta[-1],
        "verified_on_plate": bool(np.any(verified_on_plate)),
        "max_recorded_failure_run": longest_true_run(
            recorded_placement_failures
        ),
        "ever_follows_eef": any(scene["bowl_follows_eef"] for scene in scenes),
        "ever_held": any(scene["held"] for scene in scenes),
        "ever_on_plate": any(scene["on_plate"] for scene in scenes),
        "transitions": stage_transitions(scenes),
    }

    if first_close is not None:
        scene = scenes[first_close]
        close_eef_bowl_delta = (
            np.asarray(scene["positions"]["eef"])
            - np.asarray(scene["positions"]["target_bowl"])
        )
        result.update(
            {
                "close_eef_bowl_distance": scene["eef_bowl_distance"],
                "close_eef_bowl_delta": close_eef_bowl_delta,
                "close_bowl_lift": scene["bowl_lift"],
                "close_stage": scene["stage"],
            }
        )

        end = min(first_close + 25, len(scenes))
        window = scenes[first_close:end]
        result["post_close_max_lift_25"] = max(scene["bowl_lift"] for scene in window)
        result["post_close_follows_25"] = any(
            scene["bowl_follows_eef"] for scene in window
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace_dir",
        nargs="?",
        default="task5_scene_trace_baseline_v2/libero_spatial_5",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(str(Path(args.trace_dir) / "episode_*.jsonl")))
    if not paths:
        raise SystemExit(f"No episode_*.jsonl files found under {args.trace_dir}")

    for path in paths:
        result = summarize(path)
        print(f"\n{result['file']}")
        print(
            f"success={result['success']} steps={result['steps']} "
            f"first_close={result['first_close']}"
        )
        if result["first_close"] is not None:
            print(
                "at_close: "
                f"eef_bowl={result['close_eef_bowl_distance']:.4f}m "
                f"delta={np.round(result['close_eef_bowl_delta'], 4).tolist()} "
                f"lift={result['close_bowl_lift']:.4f}m "
                f"stage={result['close_stage']}"
            )
            print(
                "within_25_steps: "
                f"max_lift={result['post_close_max_lift_25']:.4f}m "
                f"follows_eef={result['post_close_follows_25']}"
            )
        print(
            "whole_episode: "
            f"min_eef_bowl={result['min_eef_bowl_distance']:.4f}m "
            f"max_lift={result['max_bowl_lift']:.4f}m "
            f"held={result['ever_held']} on_plate={result['ever_on_plate']}"
        )
        print(
            "placement: "
            f"min_xy={result['min_bowl_plate_xy']:.4f}m "
            f"at_step={result['min_plate_step']} "
            f"delta_at_min={np.round(result['delta_at_min_plate'], 4).tolist()} "
            f"final_delta={np.round(result['final_bowl_plate_delta'], 4).tolist()} "
            f"verified={result['verified_on_plate']} "
            f"max_failure_run={result['max_recorded_failure_run']}"
        )
        print(f"stages: {result['transitions']}")


if __name__ == "__main__":
    main()

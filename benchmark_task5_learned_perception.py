"""Benchmark persistent learned perception on a LIBERO task-5 reset frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_mask_perception import Task5LearnedMaskPerception
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv


OUT = Path("vision_debug/learned_perception_benchmark.json")
RUNS = 5


def main():
    cfg = LiberoEnv(task="libero_spatial", task_ids=[5])
    all_envs = make_env(cfg, n_envs=1, use_async_envs=False)
    container = all_envs["libero_spatial"]
    env = next(iter(container.values())) if isinstance(container, dict) else container[0]

    try:
        obs, _ = env.reset()
        image = np.asarray(obs["pixels"]["image"])[0]
        perception = Task5LearnedMaskPerception()
        print(f"model load: {perception.load_seconds:.2f}s")

        results = []
        for run in range(RUNS):
            prediction = perception.predict(image)
            counts = {}
            for item in prediction["detections"]:
                counts[item["label"]] = counts.get(item["label"], 0) + 1
            timing = prediction["timing_ms"]
            row = {
                "run": run,
                "counts": counts,
                "timing_ms": timing,
            }
            results.append(row)
            print(
                f"run={run} counts={counts} "
                f"detector={timing['detector']:.1f}ms "
                f"sam={timing['sam']:.1f}ms "
                f"total={timing['total']:.1f}ms"
            )

        warm = results[1:]
        totals = np.asarray([row["timing_ms"]["total"] for row in warm])
        detector = np.asarray(
            [row["timing_ms"]["detector"] for row in warm]
        )
        sam = np.asarray([row["timing_ms"]["sam"] for row in warm])
        summary = {
            "model_load_seconds": perception.load_seconds,
            "runs": results,
            "warm_runs": len(warm),
            "detector_median_ms": float(np.median(detector)),
            "sam_median_ms": float(np.median(sam)),
            "total_median_ms": float(np.median(totals)),
            "total_max_ms": float(np.max(totals)),
        }
        if perception.device.startswith("cuda"):
            import torch

            summary["cuda_peak_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "warm median: "
            f"detector={summary['detector_median_ms']:.1f}ms "
            f"sam={summary['sam_median_ms']:.1f}ms "
            f"total={summary['total_median_ms']:.1f}ms"
        )
        print("report:", OUT.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()

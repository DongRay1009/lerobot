"""Generate SAM masks from Grounding DINO boxes and compare with oracle masks."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor


ORACLE_PATHS = {
    "target_bowl": "agentview_target_bowl.png",
    "distractor_bowl": "agentview_distractor_bowl.png",
    "plate": "agentview_plate.png",
    "ramekin": "agentview_ramekin.png",
}

COLORS = [
    np.array([255, 70, 70], dtype=np.float32),
    np.array([255, 170, 60], dtype=np.float32),
    np.array([70, 255, 130], dtype=np.float32),
    np.array([70, 160, 255], dtype=np.float32),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("vision_debug/obs_image.png"),
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("vision_debug/learned_detection/detections.json"),
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=Path("vision_debug/masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vision_debug/learned_masks"),
    )
    parser.add_argument("--model", default="facebook/sam-vit-base")
    return parser.parse_args()


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def postprocess_masks(processor, pred_masks, inputs):
    method = getattr(processor, "post_process_masks", None)
    if method is None:
        method = processor.image_processor.post_process_masks
    return method(
        pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]


def mask_iou(left, right):
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def load_oracle_masks(directory):
    masks = {}
    for key, filename in ORACLE_PATHS.items():
        path = directory / filename
        if path.is_file():
            masks[key] = np.asarray(Image.open(path).convert("L")) > 0
    return masks


def match_oracle_objects(items, oracle_masks):
    for item in items:
        if item["label"] == "white plate" and "plate" in oracle_masks:
            item["matched_object"] = "plate"
        elif item["label"] == "white ramekin" and "ramekin" in oracle_masks:
            item["matched_object"] = "ramekin"

    bowls = [item for item in items if item["label"] == "black bowl"]
    bowl_keys = [
        key
        for key in ("target_bowl", "distractor_bowl")
        if key in oracle_masks
    ]
    if len(bowls) == 2 and len(bowl_keys) == 2:
        best_score = -1.0
        best_assignment = None
        for assignment in itertools.permutations(bowl_keys):
            score = sum(
                mask_iou(item["mask"], oracle_masks[key])
                for item, key in zip(bowls, assignment, strict=True)
            )
            if score > best_score:
                best_score = score
                best_assignment = assignment
        for item, key in zip(bowls, best_assignment, strict=True):
            item["matched_object"] = key

    for item in items:
        key = item.get("matched_object")
        item["oracle_iou"] = (
            mask_iou(item["mask"], oracle_masks[key])
            if key in oracle_masks
            else None
        )


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.detections.is_file():
        raise FileNotFoundError(
            f"Detection report not found: {args.detections}"
        )

    report = json.loads(args.detections.read_text())
    detections = report["detections"]
    if not detections:
        raise RuntimeError("Detection report has no selected detections")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    boxes = [item["box_xyxy"] for item in detections]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("image:", args.image.resolve())
    print("device:", device)
    print("model:", args.model)
    print("boxes:", len(boxes))

    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device).eval()
    inputs = processor(
        images=image,
        input_boxes=[boxes],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True)

    masks = postprocess_masks(processor, outputs.pred_masks, inputs)
    iou_scores = outputs.iou_scores.detach().cpu()
    if masks.ndim != 4:
        raise RuntimeError(f"Unexpected postprocessed mask shape: {masks.shape}")

    items = []
    for index, detection in enumerate(detections):
        best = int(iou_scores[0, index].argmax())
        mask = masks[index, best].detach().cpu().numpy().astype(bool)
        ys, xs = np.nonzero(mask)
        mask_box = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if len(xs)
            else None
        )
        items.append(
            {
                "index": index,
                "label": detection["label"],
                "detection_score": detection["score"],
                "detection_box_xyxy": detection["box_xyxy"],
                "sam_candidate": best,
                "sam_predicted_iou": float(iou_scores[0, index, best]),
                "pixel_count": int(mask.sum()),
                "mask_box_xyxy": mask_box,
                "mask": mask,
            }
        )

    oracle_masks = load_oracle_masks(args.oracle_dir)
    match_oracle_objects(items, oracle_masks)

    overlay = np.asarray(image).astype(np.float32).copy()
    for item, color in zip(items, COLORS, strict=True):
        mask = item["mask"]
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
        filename = f"{item['index']:02d}_{slug(item['label'])}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            args.output_dir / filename
        )
        item["mask_file"] = filename

    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_image)
    for item, color in zip(items, COLORS, strict=True):
        box = item["detection_box_xyxy"]
        display_color = tuple(int(value) for value in color)
        draw.rectangle(box, outline=display_color, width=2)
        matched = item.get("matched_object", "unmatched")
        iou = item.get("oracle_iou")
        iou_text = f" IoU={iou:.3f}" if iou is not None else ""
        draw.text(
            (box[0] + 2, max(0, box[1] - 11)),
            f"{matched}{iou_text}",
            fill=display_color,
            stroke_width=2,
            stroke_fill="black",
        )

    serializable = []
    for item in items:
        output = {key: value for key, value in item.items() if key != "mask"}
        output["sam_predicted_iou"] = round(output["sam_predicted_iou"], 6)
        if output["oracle_iou"] is not None:
            output["oracle_iou"] = round(output["oracle_iou"], 6)
        serializable.append(output)
        print(
            f"{output['index']}: {output['label']} -> "
            f"{output.get('matched_object')} "
            f"pixels={output['pixel_count']} "
            f"SAM_IoU={output['sam_predicted_iou']:.3f} "
            f"oracle_IoU={output['oracle_iou']}"
        )

    output_report = {
        "image": str(args.image),
        "detection_report": str(args.detections),
        "model": args.model,
        "masks": serializable,
    }
    report_path = args.output_dir / "mask_report.json"
    overlay_path = args.output_dir / "learned_mask_overlay.png"
    report_path.write_text(json.dumps(output_report, indent=2) + "\n")
    overlay_image.save(overlay_path)
    print("report:", report_path.resolve())
    print("overlay:", overlay_path.resolve())


if __name__ == "__main__":
    main()

"""Run open-vocabulary detection on a saved LIBERO task-5 RGB frame."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_LABELS = [
    "black bowl",
    "white plate",
    "white ramekin",
]
CLASS_THRESHOLDS = {
    "black bowl": 0.20,
    "white plate": 0.30,
    "white ramekin": 0.30,
}
EXPECTED_COUNTS = {
    "black bowl": 2,
    "white plate": 1,
    "white ramekin": 1,
}
NMS_IOU_THRESHOLD = 0.50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("vision_debug/obs_image.png"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vision_debug/learned_detection"),
    )
    parser.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    return parser.parse_args()


def postprocess(processor, outputs, inputs, text_labels, image, args):
    """Support both current and older Transformers processor signatures."""
    method = processor.post_process_grounded_object_detection
    parameters = inspect.signature(method).parameters
    common = {
        "threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "target_sizes": [(image.height, image.width)],
    }
    if "text_labels" in parameters:
        return method(outputs, text_labels=text_labels, **common)[0]
    return method(outputs, inputs.input_ids, **common)[0]


def result_labels(result):
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels", [])
    return [str(label) for label in labels]


def box_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def select_detections(detections, class_thresholds=None):
    class_thresholds = class_thresholds or CLASS_THRESHOLDS
    selected = []
    for label in DEFAULT_LABELS:
        candidates = [
            item
            for item in detections
            if item["label"] == label
            and item["score"] >= class_thresholds[label]
        ]
        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = []
        for candidate in candidates:
            if all(
                box_iou(candidate["box_xyxy"], item["box_xyxy"])
                < NMS_IOU_THRESHOLD
                for item in kept
            ):
                kept.append(candidate)
        selected.extend(kept[: EXPECTED_COUNTS[label]])
    return selected


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(args.image).convert("RGB")
    text_labels = [args.labels]

    print("image:", args.image.resolve())
    print("size:", image.size)
    print("device:", device)
    print("model:", args.model)
    print("labels:", args.labels)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model = model.to(device).eval()

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    result = postprocess(
        processor,
        outputs,
        inputs,
        text_labels,
        image,
        args,
    )
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result_labels(result)

    raw_detections = []
    colors = {
        "black bowl": "#ff4d4d",
        "white plate": "#4dff88",
        "white ramekin": "#4da6ff",
    }
    for index, (box, score, label) in enumerate(
        zip(boxes, scores, labels, strict=True)
    ):
        box = [round(float(value), 2) for value in box]
        score = float(score)
        detection = {
            "index": index,
            "label": label,
            "score": round(score, 6),
            "box_xyxy": box,
        }
        raw_detections.append(detection)

    detections = select_detections(raw_detections)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, detection in enumerate(detections):
        detection["selected_index"] = index
        box = detection["box_xyxy"]
        score = detection["score"]
        label = detection["label"]
        color = colors.get(label, "#ffd24d")
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0] + 3, max(0, box[1] - 12)),
            f"{index}: {label} {score:.3f}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
        print(
            f"{index}: label={label!r} score={score:.3f} box={box}"
        )

    report = {
        "image": str(args.image),
        "image_size": list(image.size),
        "model": args.model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "requested_labels": args.labels,
        "class_thresholds": CLASS_THRESHOLDS,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "raw_detections": raw_detections,
        "detections": detections,
    }
    report_path = args.output_dir / "detections.json"
    overlay_path = args.output_dir / "detection_overlay.png"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    overlay.save(overlay_path)

    counts = {
        label: sum(item["label"] == label for item in detections)
        for label in DEFAULT_LABELS
    }
    print("selected detections:", len(detections))
    print("selected counts:", counts)
    print("expected counts:", EXPECTED_COUNTS)
    print("report:", report_path.resolve())
    print("overlay:", overlay_path.resolve())


if __name__ == "__main__":
    main()

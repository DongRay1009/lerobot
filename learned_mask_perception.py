"""Reusable Grounding DINO + SAM perception for LIBERO task 5."""

from __future__ import annotations

import inspect
import time

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from inspect_task5_learned_detection import (
    DEFAULT_LABELS,
    select_detections,
)


class Task5LearnedMaskPerception:
    """Detect task objects from RGB and refine boxes into instance masks."""

    def __init__(
        self,
        device=None,
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
        box_threshold=0.15,
        text_threshold=0.15,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.text_labels = [DEFAULT_LABELS]

        start = time.perf_counter()
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
            .to(self.device)
            .eval()
        )
        self.sam_processor = SamProcessor.from_pretrained(segmenter_id)
        self.sam = SamModel.from_pretrained(segmenter_id).to(self.device).eval()
        self.load_seconds = time.perf_counter() - start

    def _sync(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _postprocess_detections(
        self,
        outputs,
        inputs,
        image,
        class_thresholds=None,
        box_threshold=None,
        text_threshold=None,
    ):
        method = self.detector_processor.post_process_grounded_object_detection
        parameters = inspect.signature(method).parameters
        common = {
            "threshold": (
                self.box_threshold
                if box_threshold is None
                else box_threshold
            ),
            "text_threshold": (
                self.text_threshold
                if text_threshold is None
                else text_threshold
            ),
            "target_sizes": [(image.height, image.width)],
        }
        if "text_labels" in parameters:
            result = method(
                outputs,
                text_labels=self.text_labels,
                **common,
            )[0]
        else:
            result = method(outputs, inputs.input_ids, **common)[0]

        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels", [])
        raw = []
        for index, (box, score, label) in enumerate(
            zip(
                result["boxes"].detach().cpu().tolist(),
                result["scores"].detach().cpu().tolist(),
                labels,
                strict=True,
            )
        ):
            raw.append(
                {
                    "index": index,
                    "label": str(label),
                    "score": float(score),
                    "box_xyxy": [float(value) for value in box],
                }
            )
        return (
            select_detections(raw, class_thresholds=class_thresholds),
            raw,
        )

    def _postprocess_masks(self, pred_masks, inputs):
        method = getattr(self.sam_processor, "post_process_masks", None)
        if method is None:
            method = self.sam_processor.image_processor.post_process_masks
        return method(
            pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]

    def predict(
        self,
        image,
        class_thresholds=None,
        box_threshold=None,
        text_threshold=None,
    ):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = image.convert("RGB")

        self._sync()
        total_start = time.perf_counter()
        detector_inputs = self.detector_processor(
            images=image,
            text=self.text_labels,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            detector_outputs = self.detector(**detector_inputs)
        self._sync()
        detector_end = time.perf_counter()

        detections, raw_detections = self._postprocess_detections(
            detector_outputs,
            detector_inputs,
            image,
            class_thresholds=class_thresholds,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not detections:
            return {
                "detections": [],
                "raw_detections": raw_detections,
                "timing_ms": {
                    "detector": (detector_end - total_start) * 1000,
                    "sam": 0.0,
                    "total": (time.perf_counter() - total_start) * 1000,
                },
            }

        boxes = [item["box_xyxy"] for item in detections]
        sam_start = time.perf_counter()
        sam_inputs = self.sam_processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            sam_outputs = self.sam(**sam_inputs, multimask_output=True)
        self._sync()
        sam_end = time.perf_counter()

        masks = self._postprocess_masks(sam_outputs.pred_masks, sam_inputs)
        iou_scores = sam_outputs.iou_scores.detach().cpu()
        output = []
        for index, detection in enumerate(detections):
            best = int(iou_scores[0, index].argmax())
            item = dict(detection)
            item["mask"] = (
                masks[index, best].detach().cpu().numpy().astype(bool)
            )
            item["sam_predicted_iou"] = float(
                iou_scores[0, index, best]
            )
            output.append(item)

        return {
            "detections": output,
            "raw_detections": raw_detections,
            "timing_ms": {
                "detector": (detector_end - total_start) * 1000,
                "sam": (sam_end - sam_start) * 1000,
                "total": (sam_end - total_start) * 1000,
            },
        }

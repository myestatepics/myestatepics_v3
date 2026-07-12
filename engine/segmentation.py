
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


MODEL_ID = "nvidia/segformer-b4-finetuned-ade-512-512"

PRIORITY_NAMES = [
    "wall",
    "ceiling",
    "floor",
    "window",
    "curtain",
    "door",
    "cabinet",
    "painting",
    "sofa",
    "table",
    "chair",
    "bed",
    "lamp",
    "mirror",
    "rug",
    "sky",
    "plant",
]


@dataclass
class SegmentationResult:
    label_map: np.ndarray
    class_coverage: dict[str, float]
    inference_seconds: float
    name_to_id: dict[str, int]

    def masks(self, name: str) -> np.ndarray:
        class_id = self.name_to_id.get(name.lower())
        if class_id is None:
            return np.zeros(self.label_map.shape, dtype=np.float32)

        return (self.label_map == class_id).astype(np.float32)


class SceneSegmenter:
    def __init__(self, device: str | None = None):
        if device is None:
            selected = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            selected = device

        self.device = selected
        self.processor = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self.name_to_id: dict[str, int] = {}

        print(f"Segmentation device: {self.device}")

    def _resolve_priority_classes(self) -> dict[str, int]:
        resolved: dict[str, int] = {}

        for requested in PRIORITY_NAMES:
            matches = [
                (class_id, label)
                for class_id, label in self.id2label.items()
                if requested.lower() in label.lower()
            ]

            if matches:
                resolved[requested] = int(matches[0][0])

        return resolved

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        started = time.perf_counter()

        self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(class_id): str(label)
            for class_id, label in self.model.config.id2label.items()
        }
        self.name_to_id = self._resolve_priority_classes()

        elapsed = time.perf_counter() - started

        print(
            f"Loaded {MODEL_ID} on {self.device} "
            f"in {elapsed:.2f}s"
        )
        print("Resolved classes:")
        for name, class_id in self.name_to_id.items():
            print(
                f"  {name}: {class_id} "
                f"({self.id2label[class_id]})"
            )

    def _infer(
        self,
        rgb: np.ndarray,
        device: str,
    ) -> tuple[np.ndarray, float]:
        assert self.processor is not None
        assert self.model is not None

        original_height, original_width = rgb.shape[:2]

        inputs = self.processor(
            images=rgb,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        started = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            label_map = logits.argmax(dim=1)[0]

        if device == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - started

        label_np = label_map.detach().cpu().numpy()

        max_label = int(label_np.max()) if label_np.size else 0
        dtype = np.uint8 if max_label <= 255 else np.uint16

        return label_np.astype(dtype), elapsed

    def segment(self, rgb: np.ndarray) -> SegmentationResult:
        self._load()

        try:
            label_map, elapsed = self._infer(rgb, self.device)

        except Exception as exc:
            if self.device != "mps":
                raise

            print(
                f"MPS inference failed: {exc}. "
                f"Retrying on CPU."
            )

            self.device = "cpu"
            assert self.model is not None
            self.model.to("cpu")
            label_map, elapsed = self._infer(rgb, "cpu")

        total_pixels = label_map.size
        class_coverage: dict[str, float] = {}

        class_ids, counts = np.unique(
            label_map,
            return_counts=True,
        )

        for class_id, count in zip(class_ids, counts):
            percent = float(count / total_pixels * 100.0)

            if percent <= 0.1:
                continue

            label_name = self.id2label.get(
                int(class_id),
                f"class_{int(class_id)}",
            )
            class_coverage[label_name] = percent

        return SegmentationResult(
            label_map=label_map,
            class_coverage=class_coverage,
            inference_seconds=elapsed,
            name_to_id=self.name_to_id.copy(),
        )

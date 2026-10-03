"""Обёртка над YOLO: единый результат для classify- и detect-моделей."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .hazards import get_hazard, normalize


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    label: str


@dataclass
class FrameResult:
    """Вероятности классов для одного кадра (+ рамки, если модель detect)."""

    scores: dict[str, float]
    boxes: list[Box] = field(default_factory=list)

    def top(self) -> tuple[str, float]:
        if not self.scores:
            return "normal", 0.0
        label = max(self.scores, key=lambda k: (self.scores[k], get_hazard(k).severity))
        return label, self.scores[label]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def parse_result(result: Any, box_conf: float = 0.35) -> FrameResult:
    """Превращает `ultralytics.engine.results.Results` в `FrameResult`."""
    names = {int(k): normalize(v) for k, v in dict(result.names).items()}

    probs = getattr(result, "probs", None)
    if probs is not None:
        raw = _to_numpy(probs.data).astype(float).ravel()
        return FrameResult({names[i]: float(raw[i]) for i in names if i < len(raw)})

    boxes_obj = getattr(result, "boxes", None)
    scores = {name: 0.0 for name in names.values()}
    boxes: list[Box] = []
    if boxes_obj is not None and len(boxes_obj) > 0:
        confs = _to_numpy(boxes_obj.conf).ravel()
        clss = _to_numpy(boxes_obj.cls).ravel().astype(int)
        xyxy = _to_numpy(boxes_obj.xyxy).reshape(-1, 4)
        for conf, cls_id, (x1, y1, x2, y2) in zip(confs, clss, xyxy, strict=False):
            if conf < box_conf:
                continue
            label = names.get(int(cls_id), f"class_{cls_id}")
            boxes.append(Box(int(x1), int(y1), int(x2), int(y2), float(conf), label))
            scores[label] = max(scores.get(label, 0.0), float(conf))
    # У detect-модели нет класса «норма» — выводим его как отсутствие угрозы.
    danger_max = max((v for k, v in scores.items() if get_hazard(k).danger), default=0.0)
    scores["normal"] = max(scores.get("normal", 0.0), 1.0 - danger_max)
    return FrameResult(scores, boxes)


class Detector:
    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        from ultralytics import YOLO  # тяжёлый импорт — только когда реально нужен

        self.path = model_path
        self.model = YOLO(str(model_path))
        self.task: str = self.model.task
        self.names: list[str] = [normalize(n) for n in self.model.names.values()]
        self.device = device
        self._lock = threading.Lock()

    def predict(self, frame: np.ndarray, box_conf: float = 0.35) -> FrameResult:
        with self._lock:
            kwargs: dict[str, Any] = {"verbose": False}
            if self.device:
                kwargs["device"] = self.device
            if self.task == "detect":
                kwargs["conf"] = box_conf
            result = self.model(frame, **kwargs)[0]
        return parse_result(result, box_conf)

"""Отрисовка рамок и статуса поверх кадра (с поддержкой кириллицы)."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import Box
from .hazards import get_hazard
from .tracker import Decision

_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _corners(img: np.ndarray, b: Box, color: tuple[int, int, int]) -> None:
    length = max(6, min(24, (b.x2 - b.x1) // 4, (b.y2 - b.y1) // 4))
    for (x, y), (dx, dy) in (
        ((b.x1, b.y1), (1, 1)),
        ((b.x2, b.y1), (-1, 1)),
        ((b.x1, b.y2), (1, -1)),
        ((b.x2, b.y2), (-1, -1)),
    ):
        cv2.line(img, (x, y), (x + dx * length, y), color, 3, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * length), color, 3, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    decision: Decision,
    boxes: list[Box] | None = None,
    source: str = "",
    fps: float | None = None,
    recording: bool = False,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.6, min(2.0, w / 960))
    fs = int(15 * scale)
    bar_h = int(34 * scale)
    hz = get_hazard(decision.label)
    texts: list[tuple[tuple[int, int], str, tuple[int, int, int], int]] = []

    for b in boxes or []:
        bh = get_hazard(b.label)
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), bh.bgr, 1, cv2.LINE_AA)
        _corners(img, b, bh.bgr)
        text = f"{bh.title} {b.conf * 100:.0f}%"
        tw = int(_font(fs).getlength(text))
        ty = max(0, b.y1 - fs - 8)
        cv2.rectangle(img, (b.x1, ty), (b.x1 + tw + 10, ty + fs + 8), bh.bgr, -1)
        texts.append(((b.x1 + 5, ty + 3), text, (255, 255, 255), fs))

    # Полупрозрачная верхняя панель статуса
    top = img[:bar_h].astype(np.float32)
    img[:bar_h] = (top * 0.35).astype(np.uint8)
    dot = (60, 60, 235) if decision.active else ((40, 190, 235) if decision.frame_danger else (80, 200, 80))
    cv2.circle(img, (int(16 * scale), bar_h // 2), int(6 * scale), dot, -1, cv2.LINE_AA)

    status = f"{hz.title.upper()} · {decision.confidence * 100:.0f}%"
    if not decision.active and decision.frame_danger:
        status = "ПРОВЕРКА…"
    texts.append(((int(30 * scale), bar_h // 2 - fs // 2 - 2), status, (235, 235, 235), fs))
    right = datetime.now().strftime("%H:%M:%S")
    if fps is not None:
        right = f"{fps:.1f} FPS · {right}"
    if source:
        right = f"{source} · {right}"
    rw = int(_font(fs - 2).getlength(right))
    texts.append(((w - rw - 12, bar_h // 2 - fs // 2), right, (170, 170, 170), fs - 2))

    if recording:
        cv2.circle(img, (w - 18, bar_h + 18), 7, (40, 40, 230), -1, cv2.LINE_AA)
        texts.append(((w - 58, bar_h + 9), "REC", (40, 40, 230), fs - 3))

    if decision.active and not boxes:
        cv2.rectangle(img, (2, bar_h + 2), (w - 3, h - 3), hz.bgr, max(2, int(3 * scale)))

    # Весь текст — за одно преобразование в PIL (раньше было по одному на каждую надпись).
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for pos, text, (b, g, r), size in texts:
        draw.text(pos, text, font=_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

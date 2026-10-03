"""Запись видеоклипов инцидентов с предзаписью (несколько секунд до тревоги)."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ClipRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        fps: float = 10.0,
        pre_seconds: float = 3.0,
        post_seconds: float = 10.0,
        max_seconds: float = 300.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.fps = max(1.0, float(fps))
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.max_seconds = max_seconds
        # Буфер с запасом на 30 FPS; при старте записи берём из него последние pre_seconds.
        self._buffer: deque[np.ndarray] = deque(maxlen=max(1, int(30 * pre_seconds)))
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._path: Path | None = None
        self._started = 0.0
        self._last_active = 0.0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def set_fps(self, fps: float) -> None:
        """FPS клипа = реальная скорость обработки; меняется только между записями."""
        if self._writer is None and fps > 0:
            self.fps = max(1.0, min(30.0, float(fps)))

    def _open(self, frame: np.ndarray) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        self._size = (w, h)
        self._path = self.out_dir / f"incident_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self._writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Не удалось открыть файл записи {self._path}")

    def _write(self, frame: np.ndarray) -> None:
        assert self._writer is not None and self._size is not None
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)

    def push(self, frame: np.ndarray, active: bool, now: float | None = None) -> Path | None:
        """Добавляет кадр. Возвращает путь к клипу, когда тот закончен."""
        now = time.monotonic() if now is None else now
        if active:
            self._last_active = now
            if self._writer is None:
                self._open(frame)
                self._started = now
                keep = int(self.fps * self.pre_seconds)
                for old in list(self._buffer)[-keep:] if keep > 0 else []:
                    self._write(old)
                self._buffer.clear()
        if self._writer is None:
            self._buffer.append(frame)
            return None
        self._write(frame)
        if now - self._last_active > self.post_seconds or now - self._started > self.max_seconds:
            return self.close()
        return None

    def close(self) -> Path | None:
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        path, self._path = self._path, None
        return path

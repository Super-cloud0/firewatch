"""Источники кадров: веб-камера, IP-камера (RTSP/HTTP), видеофайл, экран."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class FrameSource(ABC):
    live: bool = True
    fps: float = 0.0

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """Возвращает следующий кадр (BGR) или None, если кадров больше нет/нет связи."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Видеофайл читается последовательно — каждый кадр будет проанализирован."""

    live = False

    def __init__(self, path: str, frame_step: int = 1) -> None:
        self.path = path
        self.frame_step = max(1, frame_step)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.path}")
        self.fps = (self._cap.get(cv2.CAP_PROP_FPS) or 25.0) / self.frame_step

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        assert self._cap is not None
        for _ in range(self.frame_step - 1):
            if not self._cap.grab():
                return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class StreamSource(FrameSource):
    """Камера/поток. Кадры читаются фоновым потоком, отдаётся всегда самый свежий —
    поэтому медленная модель не создаёт растущую задержку."""

    live = True

    def __init__(self, target: int | str, reconnect: bool = True) -> None:
        self.target = target
        self.reconnect = reconnect
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._read_seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def _connect(self) -> bool:
        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return True

    def open(self) -> None:
        if not self._connect():
            raise RuntimeError(f"Не удалось подключиться к источнику: {self.target}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._grab_loop, name="frame-grabber", daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._cap
            ok, frame = cap.read() if cap is not None else (False, None)
            if ok:
                backoff = 1.0
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
                continue
            if not self.reconnect:
                break
            self.error = "Потеряна связь с камерой, переподключение…"
            if cap is not None:
                cap.release()
            self._cap = None
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 30.0)
            if self._connect():
                self.error = None
        with self._cond:
            self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == self._read_seq and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (self._thread is not None and not self._thread.is_alive()):
                    return None
                self._cond.wait(remaining)
            if self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._frame

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ScreenSource(FrameSource):
    live = True

    def __init__(self, monitor: int = 1, max_fps: float = 10.0) -> None:
        self.monitor = monitor
        self.fps = max_fps
        self._sct = None
        self._last = 0.0

    def open(self) -> None:
        from mss import mss

        self._sct = mss()
        if self.monitor >= len(self._sct.monitors):
            raise RuntimeError(f"Монитор {self.monitor} не найден")

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        assert self._sct is not None
        wait = 1.0 / self.fps - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        img = np.asarray(self._sct.grab(self._sct.monitors[self.monitor]))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


@dataclass
class SourceSpec:
    kind: str  # "webcam" | "stream" | "file" | "screen"
    value: str | int
    label: str

    def build(self) -> FrameSource:
        if self.kind == "webcam":
            return StreamSource(int(self.value))
        if self.kind == "stream":
            return StreamSource(str(self.value))
        if self.kind == "file":
            return VideoFileSource(str(self.value))
        if self.kind == "screen":
            return ScreenSource(int(self.value))
        raise ValueError(f"Неизвестный тип источника: {self.kind}")


def parse_source(value: str) -> SourceSpec:
    """Для CLI: '0' → веб-камера, 'rtsp://…' → поток, 'screen' / 'screen:2' → экран, иначе — файл."""
    v = value.strip()
    if v.isdigit():
        return SourceSpec("webcam", int(v), f"Камера {v}")
    if v.startswith("screen"):
        idx = int(v.split(":", 1)[1]) if ":" in v else 1
        return SourceSpec("screen", idx, f"Экран {idx}")
    if "://" in v:
        return SourceSpec("stream", v, v.split("@")[-1])
    return SourceSpec("file", v, Path(v).name)

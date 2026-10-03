"""Фоновый мониторинг: источник → модель → трекер → журнал/уведомления/запись.

Работает в отдельном потоке, поэтому интерфейс Streamlit остаётся отзывчивым,
а мониторинг не прерывается при переключении вкладок или перезагрузке страницы.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionParams, Settings
from .detector import Detector
from .notifier import AsyncSender, TelegramNotifier, format_alert, format_incident_closed
from .overlay import draw_overlay
from .recorder import ClipRecorder
from .sources import SourceSpec
from .storage import EventStore
from .tracker import Decision, ThreatTracker

log = logging.getLogger(__name__)


PREVIEW_MAX_WIDTH = 960


def encode_preview(frame: np.ndarray, max_width: int = PREVIEW_MAX_WIDTH) -> bytes | None:
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


@dataclass
class MonitorOptions:
    source: SourceSpec
    params: DetectionParams = field(default_factory=DetectionParams)
    notify: bool = True
    notify_photo: bool = True
    notify_on_clear: bool = False
    notify_cooldown: float = 60.0  # мин. секунд между тревогами в Telegram
    record: bool = False
    record_pre: float = 3.0
    record_post: float = 10.0
    lat: float | None = None
    lon: float | None = None


@dataclass
class MonitorState:
    running: bool = False
    finished: bool = False
    error: str | None = None
    source: str = ""
    started_at: datetime | None = None
    frame: np.ndarray | None = None
    preview: bytes | None = None  # JPEG для интерфейса, кодируется один раз в рабочем потоке
    decision: Decision | None = None
    fps: float = 0.0
    frames: int = 0
    incidents: int = 0
    incident_id: int | None = None
    recording: bool = False


class Monitor:
    def __init__(
        self,
        detector: Detector,
        store: EventStore,
        settings: Settings,
        options: MonitorOptions,
        notifier: TelegramNotifier | None = None,
        sender: AsyncSender | None = None,
    ) -> None:
        self.detector = detector
        self.store = store
        self.settings = settings
        self.options = options
        self.notifier = notifier
        self.sender = sender
        self.tracker = ThreatTracker(options.params)
        self._state = MonitorState(source=options.source.label)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ecowatch-monitor", daemon=True)
        self._incident: dict | None = None
        self._last_alert = float("-inf")

    # ---------- управление ----------
    def start(self) -> None:
        with self._lock:
            self._state.running = True
            self._state.started_at = datetime.now()
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)

    def update_params(self, params: DetectionParams) -> None:
        """Пороги можно менять на лету, не перезапуская мониторинг."""
        self.tracker.params = params

    def state(self) -> MonitorState:
        with self._lock:
            return replace(self._state)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _set(self, **kw: object) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._state, k, v)

    # ---------- уведомления ----------
    def _notify(self, text: str, image: np.ndarray | None) -> None:
        if not (self.options.notify and self.notifier and self.notifier.enabled and self.sender):
            return
        if image is not None and self.options.notify_photo:
            self.sender.submit(self.notifier.send_photo, image, text)
        else:
            self.sender.submit(self.notifier.send_message, text)

    # ---------- инциденты ----------
    def _on_onset(self, d: Decision, frame: np.ndarray) -> None:
        now = datetime.now()
        snap: Path | None = self.settings.snapshots_dir / f"{now:%Y%m%d_%H%M%S}_{d.label}.jpg"
        try:
            snap.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(snap), frame):
                snap = None
        except (OSError, cv2.error):
            snap = None
        incident_id = self.store.open_incident(
            d.label,
            d.confidence,
            self.options.source.label,
            started_at=now,
            snapshot=str(snap) if snap else None,
            lat=self.options.lat,
            lon=self.options.lon,
        )
        self._incident = {"id": incident_id, "label": d.label, "peak": d.confidence, "started": now}
        log.warning("Инцидент #%s: %s (%.0f%%)", incident_id, d.label, d.confidence * 100)
        if time.monotonic() - self._last_alert >= self.options.notify_cooldown:
            self._last_alert = time.monotonic()
            self._notify(format_alert(d.label, d.confidence, self.options.source.label, now), frame)

    def _close_incident(self) -> None:
        inc = self._incident
        if inc is None:
            return
        ended = datetime.now()
        self.store.update_incident(inc["id"], ended_at=ended, peak_conf=inc["peak"], label=inc["label"])
        log.info("Инцидент #%s завершён", inc["id"])
        if self.options.notify_on_clear:
            self._notify(format_incident_closed(inc["label"], inc["peak"], inc["started"], ended), None)
        self._incident = None

    def _attach_clip(self, path: Path | None, incident_id: int | None) -> None:
        if path and incident_id is not None:
            self.store.update_incident(incident_id, clip=str(path))

    # ---------- основной цикл ----------
    def _run(self) -> None:
        source = self.options.source.build()
        recorder = (
            ClipRecorder(
                self.settings.clips_dir, pre_seconds=self.options.record_pre, post_seconds=self.options.record_post
            )
            if self.options.record
            else None
        )
        clip_incident: int | None = None
        fps = 0.0
        stamps: deque[float] = deque(maxlen=30)  # FPS по скользящему окну последних кадров
        try:
            source.open()
            if recorder and not source.live:
                recorder.set_fps(source.fps)
            while not self._stop.is_set():
                frame = source.read(timeout=5.0)
                if frame is None:
                    if source.live:
                        self._set(error=getattr(source, "error", None) or "Нет кадров от источника…")
                        continue
                    self._set(finished=True)
                    break

                result = self.detector.predict(frame, self.tracker.params.box_conf)
                d = self.tracker.update(result)

                stamps.append(time.monotonic())
                if len(stamps) > 1:
                    fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6)

                annotated = draw_overlay(
                    frame,
                    d,
                    result.boxes,
                    self.options.source.label,
                    fps,
                    recording=bool(recorder and recorder.recording),
                )

                if d.onset:
                    self._on_onset(d, annotated)
                if self._incident is not None and d.active:
                    self._incident["peak"] = max(self._incident["peak"], d.confidence)
                    self._incident["label"] = d.label
                if recorder:
                    if source.live:
                        recorder.set_fps(fps)
                    if d.active and not recorder.recording and self._incident:
                        clip_incident = self._incident["id"]
                    self._attach_clip(recorder.push(annotated, d.active), clip_incident)
                if d.cleared:
                    self._close_incident()

                preview = encode_preview(annotated)
                with self._lock:
                    s = self._state
                    s.frame, s.preview, s.decision, s.fps, s.error = annotated, preview, d, fps, None
                    s.frames += 1
                    s.incidents += int(d.onset)
                    s.incident_id = self._incident["id"] if self._incident else None
                    s.recording = bool(recorder and recorder.recording)
        except Exception as exc:  # noqa: BLE001 — ошибку показываем в интерфейсе
            log.exception("Мониторинг остановлен из-за ошибки")
            self._set(error=str(exc))
        finally:
            self._close_incident()
            if recorder:
                self._attach_clip(recorder.close(), clip_incident)
            source.close()
            self._set(running=False, recording=False)

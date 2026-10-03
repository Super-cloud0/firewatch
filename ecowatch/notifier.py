"""Уведомления в Telegram. Отправка идёт в фоновом потоке и не тормозит видео."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import datetime
from html import escape
from typing import Any

import cv2
import numpy as np
import requests

from .hazards import get_hazard

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
CAPTION_LIMIT = 1024


class TelegramNotifier:
    def __init__(self, token: str | None, chat_id: str | None, session: Any = None, timeout: float = 15) -> None:
        self.token = token
        self.chat_id = chat_id
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _call(self, method: str, **kwargs: Any) -> bool:
        if not self.enabled:
            return False
        try:
            resp = self.session.post(API.format(token=self.token, method=method), timeout=self.timeout, **kwargs)
            ok = bool(resp.json().get("ok"))
            if not ok:
                log.warning("Telegram %s: %s", method, resp.text[:300])
            return ok
        except (requests.RequestException, ValueError) as exc:
            log.warning("Telegram %s не отправлен: %s", method, exc)
            return False

    def send_message(self, text: str) -> bool:
        return self._call("sendMessage", json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})

    def send_photo(self, image_bgr: np.ndarray, caption: str = "") -> bool:
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return self.send_message(caption)
        if len(caption) > CAPTION_LIMIT:
            # Подпись к фото ограничена — длинный текст отправляем отдельным сообщением.
            return self.send_message(caption) and self.send_photo(image_bgr)
        return self._call(
            "sendPhoto",
            data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
        )


class AsyncSender:
    """Очередь задач отправки с одним рабочим потоком."""

    def __init__(self, maxsize: int = 32) -> None:
        self._queue: queue.Queue[tuple[Callable[..., bool], tuple[Any, ...]]] = queue.Queue(maxsize)
        self.sent = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="telegram-sender", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[..., bool], *args: Any) -> bool:
        try:
            self._queue.put_nowait((fn, args))
            return True
        except queue.Full:
            self.failed += 1
            return False

    def _run(self) -> None:
        while True:
            fn, args = self._queue.get()
            try:
                if fn(*args):
                    self.sent += 1
                else:
                    self.failed += 1
            except Exception:  # noqa: BLE001 — поток отправки не должен падать
                log.exception("Ошибка при отправке уведомления")
                self.failed += 1
            finally:
                self._queue.task_done()

    def join(self) -> None:
        self._queue.join()


def format_alert(label: str, conf: float, source: str, when: datetime | None = None) -> str:
    h = get_hazard(label)
    when = when or datetime.now()
    return (
        f"🚨 <b>ЭКОЛОГИЧЕСКАЯ УГРОЗА</b>\n\n"
        f"{h.icon} <b>{escape(h.title.upper())}</b>\n"
        f"{escape(h.description)}\n\n"
        f"📊 Вероятность: <b>{conf * 100:.0f}%</b>\n"
        f"📹 Источник: {escape(source)}\n"
        f"🕐 {when:%d.%m.%Y %H:%M:%S}\n\n"
        f"<i>— EcoWatch AI</i>"
    )


def format_incident_closed(label: str, peak: float, started: datetime, ended: datetime) -> str:
    h = get_hazard(label)
    minutes, seconds = divmod(int((ended - started).total_seconds()), 60)
    return (
        f"✅ <b>Угроза больше не наблюдается</b>\n\n"
        f"{h.icon} {escape(h.title)} · пик {peak * 100:.0f}%\n"
        f"⏱ Длительность: {minutes} мин {seconds} с\n"
        f"🕐 {ended:%d.%m.%Y %H:%M:%S}"
    )


def format_daily_report(summary: dict[str, Any]) -> str:
    by_label: dict[str, int] = summary.get("by_label", {})
    lines = [f"📊 <b>ОТЧЁТ ЗА {summary['day']:%d.%m.%Y}</b>", ""]
    if by_label:
        for label, n in sorted(by_label.items(), key=lambda kv: -kv[1]):
            h = get_hazard(label)
            lines.append(f"{h.icon} {escape(h.title)}: <b>{n}</b>")
        lines += ["", f"📋 Всего инцидентов: <b>{summary['total']}</b>"]
        lines.append(f"📈 Максимальная вероятность: <b>{summary['peak_conf'] * 100:.0f}%</b>")
    else:
        lines.append("Инцидентов не зафиксировано.")
    dangerous = sum(n for label, n in by_label.items() if get_hazard(label).danger)
    lines += ["", "⚠️ Повышенная активность!" if dangerous > 5 else "✅ Ситуация стабильна", "", "<i>— EcoWatch AI</i>"]
    return "\n".join(lines)

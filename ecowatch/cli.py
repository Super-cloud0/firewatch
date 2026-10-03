"""Консольный режим без браузера — для круглосуточной работы на сервере.

Примеры:
    python -m ecowatch 0                          # веб-камера 0
    python -m ecowatch rtsp://user:pass@ip/stream --record
    python -m ecowatch video.mp4 --show           # с окном предпросмотра
    python -m ecowatch report                     # отправить отчёт за сегодня в Telegram
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .config import DetectionParams, Settings
from .notifier import AsyncSender, TelegramNotifier, format_daily_report
from .sources import parse_source
from .storage import EventStore


def build_parser() -> argparse.ArgumentParser:
    d = DetectionParams()
    p = argparse.ArgumentParser(prog="ecowatch", description="EcoWatch AI — мониторинг пожаров и дыма")
    p.add_argument("source", help="0 (камера), rtsp://…, путь к видео, screen[:N] или 'report'")
    p.add_argument("--model", help="путь к весам YOLO (по умолчанию models/best.pt)")
    p.add_argument("--conf", type=float, default=d.conf_threshold, help="порог вероятности")
    p.add_argument("--margin", type=float, default=d.margin, help="отрыв от второго класса")
    p.add_argument("--confirm", type=int, default=d.confirm_frames, help="кадров для подтверждения")
    p.add_argument("--clear", type=int, default=d.clear_frames, help="кадров для отбоя")
    p.add_argument("--record", action="store_true", help="записывать клипы инцидентов")
    p.add_argument("--no-telegram", action="store_true", help="не отправлять уведомления")
    p.add_argument("--cooldown", type=float, default=60, help="пауза между тревогами, с")
    p.add_argument("--show", action="store_true", help="показывать окно с видео")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = Settings.from_env()
    if args.model:
        from pathlib import Path

        settings.model_path = Path(args.model)
    settings.ensure_dirs()
    store = EventStore(settings.db_path)
    notifier = TelegramNotifier(settings.telegram_token, settings.telegram_chat_id)

    if args.source == "report":
        if not notifier.enabled:
            logging.error("Telegram не настроен (ECOWATCH_TG_TOKEN / ECOWATCH_TG_CHAT_ID)")
            return 1
        return 0 if notifier.send_message(format_daily_report(store.day_summary())) else 1

    from .detector import Detector
    from .monitor import Monitor, MonitorOptions

    detector = Detector(settings.model_path, settings.device)
    logging.info("Модель %s (%s): %s", settings.model_path.name, detector.task, ", ".join(detector.names))
    options = MonitorOptions(
        source=parse_source(args.source),
        params=DetectionParams(
            conf_threshold=args.conf, margin=args.margin, confirm_frames=args.confirm, clear_frames=args.clear
        ),
        notify=not args.no_telegram and notifier.enabled,
        notify_cooldown=args.cooldown,
        record=args.record,
        lat=settings.camera_lat,
        lon=settings.camera_lon,
    )
    if not args.no_telegram and not notifier.enabled:
        logging.warning("Telegram не настроен — уведомления отключены")
    sender = AsyncSender()
    monitor = Monitor(detector, store, settings, options, notifier, sender)

    signal.signal(signal.SIGINT, lambda *_: monitor.stop(wait=False))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: monitor.stop(wait=False))

    monitor.start()
    logging.info("Мониторинг запущен: %s (Ctrl+C — остановить)", options.source.label)
    last_log = 0.0
    while monitor.alive:
        s = monitor.state()
        if args.show and s.frame is not None:
            import cv2

            cv2.imshow("EcoWatch AI", s.frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                monitor.stop(wait=False)
        else:
            time.sleep(0.2)
        if time.monotonic() - last_log > 30:
            last_log = time.monotonic()
            logging.info("кадров: %d · %.1f FPS · инцидентов: %d", s.frames, s.fps, s.incidents)
    monitor.stop()
    sender.join()
    s = monitor.state()
    if s.error:
        logging.error("Ошибка: %s", s.error)
    logging.info("Готово. Кадров: %d, инцидентов: %d", s.frames, s.incidents)
    return 1 if s.error else 0


if __name__ == "__main__":
    sys.exit(main())

from datetime import date, datetime

import numpy as np

from ecowatch.notifier import AsyncSender, TelegramNotifier, format_alert, format_daily_report


class FakeResp:
    def __init__(self, ok=True):
        self._ok = ok
        self.text = ""

    def json(self):
        return {"ok": self._ok}


class FakeSession:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return FakeResp(self.ok)


def test_disabled_without_credentials():
    s = FakeSession()
    n = TelegramNotifier(None, "1", session=s)
    assert not n.enabled and not n.send_message("hi") and s.calls == []


def test_send_message_and_photo():
    s = FakeSession()
    n = TelegramNotifier("T", "42", session=s)
    assert n.send_message("hello")
    assert n.send_photo(np.zeros((10, 10, 3), np.uint8), "cap")
    assert s.calls[0][0].endswith("/botT/sendMessage")
    assert s.calls[1][0].endswith("/sendPhoto") and s.calls[1][1]["data"]["chat_id"] == "42"


def test_long_caption_split():
    s = FakeSession()
    TelegramNotifier("T", "1", session=s).send_photo(np.zeros((4, 4, 3), np.uint8), "x" * 2000)
    assert [c[0].rsplit("/", 1)[1] for c in s.calls] == ["sendMessage", "sendPhoto"]


def test_async_sender_counts():
    sender = AsyncSender()
    sender.submit(lambda: True)
    sender.submit(lambda: False)
    sender.submit(lambda: 1 / 0)
    sender.join()
    assert sender.sent == 1 and sender.failed == 2


def test_format_alert_escapes_html():
    text = format_alert("fire", 0.876, "<cam>", datetime(2026, 1, 1, 9, 5))
    assert "ПОЖАР" in text and "88%" in text and "&lt;cam&gt;" in text and "01.01.2026 09:05:00" in text


def test_daily_report():
    text = format_daily_report(
        {"day": date(2026, 1, 1), "total": 3, "by_label": {"fire": 2, "smoke": 1}, "peak_conf": 0.9}
    )
    assert "Пожар: <b>2</b>" in text and "Всего инцидентов: <b>3</b>" in text and "стабильна" in text
    assert "не зафиксировано" in format_daily_report(
        {"day": date(2026, 1, 1), "total": 0, "by_label": {}, "peak_conf": 0}
    )

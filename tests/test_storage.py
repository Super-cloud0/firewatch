import json
from datetime import date, datetime

import pytest

from ecowatch.storage import EventStore


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def test_open_update_list(store):
    i = store.open_incident("fire", 0.8, "Камера 0", started_at=datetime(2026, 1, 2, 10, 0, 0), lat=51.1, lon=71.4)
    store.update_incident(i, ended_at=datetime(2026, 1, 2, 10, 1, 0), peak_conf=0.93, clip="/x.mp4")
    (row,) = store.list()
    assert row["label"] == "fire" and row["peak_conf"] == 0.93
    assert row["ended_at"] == "2026-01-02T10:01:00" and row["clip"] == "/x.mp4" and row["lat"] == 51.1


def test_update_rejects_unknown_fields(store):
    i = store.open_incident("smoke", 0.7, "s")
    with pytest.raises(ValueError):
        store.update_incident(i, id=5)


def test_list_order_since_limit_and_clear(store):
    for h in (8, 9, 10):
        store.open_incident("smoke", 0.7, "s", started_at=datetime(2026, 1, 2, h))
    assert [r["started_at"][11:13] for r in store.list()] == ["10", "09", "08"]
    assert len(store.list(since=datetime(2026, 1, 2, 9))) == 2
    assert len(store.list(limit=1)) == 1
    store.clear()
    assert store.count() == 0


def test_day_summary(store):
    store.open_incident("fire", 0.9, "s", started_at=datetime(2026, 3, 1, 12))
    store.open_incident("fire", 0.7, "s", started_at=datetime(2026, 3, 1, 13))
    store.open_incident("smoke", 0.6, "s", started_at=datetime(2026, 3, 1, 14))
    store.open_incident("fire", 0.99, "s", started_at=datetime(2026, 3, 2, 1))
    s = store.day_summary(date(2026, 3, 1))
    assert s["total"] == 3 and s["by_label"] == {"fire": 2, "smoke": 1} and s["peak_conf"] == 0.9


def test_import_legacy_json(store, tmp_path):
    legacy = tmp_path / "detection_history.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "time": "2025-05-01 12:00:00",
                    "type": "ПОЖАР",
                    "label": "fire",
                    "confidence": 87.5,
                    "source": "Веб-камера",
                },
                {"time": "broken"},
            ]
        ),
        encoding="utf-8",
    )
    assert store.import_legacy_json(legacy) == 1
    (row,) = store.list()
    assert row["label"] == "fire" and row["peak_conf"] == pytest.approx(0.875)

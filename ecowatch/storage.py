"""Журнал инцидентов в SQLite (вместо перезаписываемого JSON-файла)."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .hazards import normalize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT    NOT NULL,
    ended_at   TEXT,
    label      TEXT    NOT NULL,
    peak_conf  REAL    NOT NULL,
    source     TEXT    NOT NULL DEFAULT '',
    snapshot   TEXT,
    clip       TEXT,
    lat        REAL,
    lon        REAL
);
CREATE INDEX IF NOT EXISTS idx_incidents_started ON incidents(started_at);
"""

COLUMNS = ("id", "started_at", "ended_at", "label", "peak_conf", "source", "snapshot", "clip", "lat", "lon")


def _ts(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def open_incident(
        self,
        label: str,
        conf: float,
        source: str,
        started_at: datetime | None = None,
        snapshot: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO incidents (started_at, label, peak_conf, source, snapshot, lat, lon)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_ts(started_at or datetime.now()), label, float(conf), source, snapshot, lat, lon),
            )
            return int(cur.lastrowid)

    def update_incident(self, incident_id: int, **fields: Any) -> None:
        allowed = {"ended_at", "label", "peak_conf", "snapshot", "clip"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Нельзя обновить поля: {unknown}")
        if not fields:
            return
        if isinstance(fields.get("ended_at"), datetime):
            fields["ended_at"] = _ts(fields["ended_at"])
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE incidents SET {assignments} WHERE id = ?", (*fields.values(), incident_id))

    def list(self, since: datetime | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM incidents"
        args: list[Any] = []
        if since:
            sql += " WHERE started_at >= ?"
            args.append(_ts(since))
        sql += " ORDER BY started_at DESC, id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0])

    def day_summary(self, day: date | None = None) -> dict[str, Any]:
        day = day or date.today()
        prefix = day.isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, peak_conf FROM incidents WHERE substr(started_at, 1, 10) = ?", (prefix,)
            ).fetchall()
        counts = Counter(r["label"] for r in rows)
        peak = max((r["peak_conf"] for r in rows), default=0.0)
        return {"day": day, "total": len(rows), "by_label": dict(counts), "peak_conf": peak}

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM incidents")

    def import_legacy_json(self, path: str | Path) -> int:
        """Переносит detection_history.json из старой версии. Возвращает число записей."""
        path = Path(path)
        if not path.is_file():
            return 0
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        rows = []
        for e in entries if isinstance(entries, list) else []:
            try:
                started = datetime.strptime(str(e["time"]), "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            conf = float(e.get("confidence", 0)) / 100.0  # в старом формате — проценты
            rows.append((_ts(started), normalize(e.get("label", "normal")), conf, str(e.get("source", ""))))
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO incidents (started_at, label, peak_conf, source) VALUES (?, ?, ?, ?)", rows
            )
        return len(rows)

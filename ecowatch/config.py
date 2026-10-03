"""Настройки приложения: пути, секреты и пороги обнаружения.

Секреты (токен Telegram) читаются только из переменных окружения или файла
`.env` в корне проекта и никогда не хранятся в коде.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Минимальный парсер .env: KEY=VALUE, комментарии через #. Не перетирает окружение."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


@dataclass
class DetectionParams:
    """Пороги принятия решения. Подобраны под classify-модель из models/best.pt."""

    conf_threshold: float = 0.65  # минимальная сглаженная вероятность опасного класса
    margin: float = 0.15  # отрыв лидера от второго класса
    box_conf: float = 0.35  # мин. уверенность рамки (только для detect-моделей)
    ema_alpha: float = 0.35  # вес нового кадра при сглаживании вероятностей
    confirm_frames: int = 4  # подряд «опасных» кадров для начала инцидента
    clear_frames: int = 15  # подряд «чистых» кадров для завершения инцидента


@dataclass
class Settings:
    model_path: Path = field(default_factory=lambda: ROOT_DIR / "models" / "best.pt")
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    camera_lat: float | None = None
    camera_lon: float | None = None
    device: str | None = None  # "cpu", "0" (GPU) и т.п.; None — автоматически

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ecowatch.db"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)

    @classmethod
    def from_env(cls, dotenv: Path | None = ROOT_DIR / ".env") -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        env = os.environ
        s = cls()
        if env.get("ECOWATCH_MODEL"):
            s.model_path = Path(env["ECOWATCH_MODEL"]).expanduser()
        if env.get("ECOWATCH_DATA_DIR"):
            s.data_dir = Path(env["ECOWATCH_DATA_DIR"]).expanduser()
        s.telegram_token = env.get("ECOWATCH_TG_TOKEN") or None
        s.telegram_chat_id = env.get("ECOWATCH_TG_CHAT_ID") or None
        s.camera_lat = _float_or_none(env.get("ECOWATCH_CAMERA_LAT"))
        s.camera_lon = _float_or_none(env.get("ECOWATCH_CAMERA_LON"))
        s.device = env.get("ECOWATCH_DEVICE") or None
        return s

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.snapshots_dir, self.clips_dir):
            d.mkdir(parents=True, exist_ok=True)

from ecowatch.config import Settings
from ecowatch.hazards import get_hazard, is_danger
from ecowatch.sources import parse_source


def test_settings_from_dotenv(tmp_path, monkeypatch):
    for k in ("ECOWATCH_TG_TOKEN", "ECOWATCH_TG_CHAT_ID", "ECOWATCH_CAMERA_LAT", "ECOWATCH_DATA_DIR"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nECOWATCH_TG_TOKEN="abc"\nexport ECOWATCH_TG_CHAT_ID=42\nECOWATCH_CAMERA_LAT=bad\n'
        f"ECOWATCH_DATA_DIR={tmp_path / 'd'}\n"
    )
    s = Settings.from_env(env)
    assert s.telegram_token == "abc" and s.telegram_chat_id == "42" and s.telegram_configured
    assert s.camera_lat is None and s.db_path == tmp_path / "d" / "ecowatch.db"


def test_parse_source():
    assert parse_source("0").kind == "webcam"
    assert parse_source("screen:2").value == 2
    s = parse_source("rtsp://u:p@10.0.0.1/live")
    assert s.kind == "stream" and "p@" not in s.label  # пароль не попадает в подписи
    assert parse_source("clip.mp4").kind == "file"


def test_hazards():
    assert is_danger("Fire") and is_danger("flame") and not is_danger("normal")
    assert get_hazard("unknown_cls").title == "Unknown_cls"

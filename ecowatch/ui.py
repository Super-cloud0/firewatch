"""Веб-интерфейс EcoWatch AI на Streamlit."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .config import ROOT_DIR, DetectionParams, Settings
from .detector import Detector
from .hazards import HAZARDS, get_hazard
from .monitor import Monitor, MonitorOptions
from .notifier import AsyncSender, TelegramNotifier, format_daily_report
from .overlay import draw_overlay
from .sources import SourceSpec
from .storage import EventStore
from .tracker import Decision, evaluate_single

CSS = """
<style>
#MainMenu, footer {visibility: hidden}
.block-container {padding-top: 1.6rem}
.ew-card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:14px 16px; margin-bottom:10px}
.ew-danger {border-left:4px solid #ef4444; background:rgba(239,68,68,.08)}
.ew-warn {border-left:4px solid #eab308; background:rgba(234,179,8,.08)}
.ew-safe {border-left:4px solid #22c55e; background:rgba(34,197,94,.06)}
.ew-title {font-weight:700; font-size:1.05rem}
.ew-muted {opacity:.65; font-size:.8rem}
.ew-bar {background:rgba(128,128,128,.18); border-radius:3px; height:5px; margin:2px 0 6px}
.ew-bar > div {height:100%; border-radius:3px}
.ew-idle {border:1px dashed rgba(128,128,128,.4); border-radius:10px; height:420px;
          display:flex; align-items:center; justify-content:center; opacity:.6}
</style>
"""

SOURCES = {"webcam": "Веб-камера", "stream": "IP-камера (RTSP/HTTP)", "file": "Видеофайл", "screen": "Захват экрана"}


# ---------- общие ресурсы (одни на весь сервер) ----------
@st.cache_resource
def get_settings() -> Settings:
    s = Settings.from_env()
    s.ensure_dirs()
    return s


@st.cache_resource(show_spinner="Загрузка модели…")
def get_detector(path: str, device: str | None) -> Detector:
    return Detector(path, device)


@st.cache_resource
def get_store(path: str) -> EventStore:
    store = EventStore(path)
    legacy = ROOT_DIR / "detection_history.json"
    if store.count() == 0 and legacy.is_file():
        store.import_legacy_json(legacy)
    return store


@st.cache_resource
def get_sender() -> AsyncSender:
    return AsyncSender()


class MonitorHolder:
    """Один активный мониторинг на сервер — переживает перезагрузку страницы."""

    def __init__(self) -> None:
        self.monitor: Monitor | None = None
        self.lock = threading.Lock()


@st.cache_resource
def get_holder() -> MonitorHolder:
    return MonitorHolder()


# ---------- вспомогательное ----------
def scores_html(scores: dict[str, float]) -> str:
    rows = []
    for name, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        h = get_hazard(name)
        rows.append(
            f"<div style='display:flex;justify-content:space-between;font-size:.82rem'>"
            f"<span>{h.icon} {h.title}</span><b>{p * 100:.1f}%</b></div>"
            f"<div class='ew-bar'><div style='width:{max(p * 100, 1):.0f}%;background:{h.color}'></div></div>"
        )
    return "".join(rows)


def status_card(d: Decision | None, subtitle: str = "") -> str:
    if d is None:
        return (
            "<div class='ew-card ew-safe'><div class='ew-title'>Система готова</div>"
            "<div class='ew-muted'>Выберите источник и запустите мониторинг</div></div>"
        )
    h = get_hazard(d.label)
    if d.active:
        cls, title, desc = "ew-danger", f"{h.icon} {h.title.upper()}", h.description
    elif d.frame_danger:
        cls, title, desc = "ew-warn", "⏳ Проверка…", "Возможная угроза, ждём подтверждения на нескольких кадрах"
    else:
        cls, title, desc = "ew-safe", "✅ Угроз не обнаружено", ""
    note = "<div class='ew-muted' style='color:#eab308'>Результат неоднозначен</div>" if d.ambiguous else ""
    return (
        f"<div class='ew-card {cls}'><div class='ew-title'>{title}</div>"
        f"<div class='ew-muted'>{desc}</div><div style='margin-top:10px'>{scores_html(d.scores)}</div>"
        f"{note}<div class='ew-muted'>{subtitle}</div></div>"
    )


def incidents_df(store: EventStore) -> pd.DataFrame:
    df = pd.DataFrame(store.list())
    if df.empty:
        return df
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["Тип"] = df["label"].map(lambda k: f"{get_hazard(k).icon} {get_hazard(k).title}")
    df["Вероятность, %"] = (df["peak_conf"] * 100).round(1)
    df["Длительность, с"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()
    return df


def existing_file(value: object) -> Path | None:
    """Путь из журнала, если файл ещё на диске (пустые значения pandas приходят как NaN)."""
    if isinstance(value, str) and value and Path(value).is_file():
        return Path(value)
    return None


def empty_state(icon: str, text: str) -> None:
    st.markdown(
        f"<div class='ew-card' style='text-align:center;padding:48px 16px'>"
        f"<div style='font-size:1.8rem;opacity:.35'>{icon}</div><div class='ew-muted'>{text}</div></div>",
        unsafe_allow_html=True,
    )


# ---------- сайдбар ----------
def sidebar(
    settings: Settings, detector: Detector, holder: MonitorHolder, store: EventStore
) -> tuple[MonitorOptions | None, DetectionParams]:
    """Рисует настройки. Возвращает (опции запуска, если нажали «Запустить»; текущие пороги)."""
    sb = st.sidebar
    logo = ROOT_DIR / "logo" / "logo_small.png"
    if logo.is_file():
        sb.image(str(logo), width=130)
    sb.markdown("### 🛡️ EcoWatch AI")
    sb.caption(f"Модель: `{detector.path.name}` · {detector.task} · классы: {', '.join(detector.names)}")

    running = holder.monitor is not None and holder.monitor.alive
    sb.subheader("Источник")
    kind = sb.selectbox("Тип", list(SOURCES), format_func=SOURCES.get, disabled=running)
    spec: SourceSpec | None = None
    if kind == "webcam":
        cam = sb.number_input("Номер камеры", 0, 10, 0, disabled=running)
        spec = SourceSpec("webcam", int(cam), f"Камера {cam}")
    elif kind == "stream":
        url = sb.text_input("URL потока", placeholder="rtsp://user:pass@192.168.1.10:554/stream", disabled=running)
        if url.strip():
            spec = SourceSpec("stream", url.strip(), url.split("@")[-1][:40])
    elif kind == "file":
        up = sb.file_uploader("Видео", type=["mp4", "avi", "mov", "mkv"], disabled=running)
        if up is not None:
            path = settings.data_dir / "uploads" / up.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != up.size:
                path.write_bytes(up.getbuffer())
            spec = SourceSpec("file", str(path), up.name)
    else:
        mon = sb.number_input("Монитор", 1, 8, 1, disabled=running)
        spec = SourceSpec("screen", int(mon), f"Экран {mon}")

    with sb.expander("Чувствительность", expanded=False):
        p = DetectionParams()
        params = DetectionParams(
            conf_threshold=st.slider(
                "Порог вероятности",
                0.3,
                0.99,
                p.conf_threshold,
                0.01,
                help="Минимальная сглаженная вероятность опасного класса",
            ),
            margin=st.slider("Отрыв от 2-го класса", 0.0, 0.6, p.margin, 0.01, help="Защита от неоднозначных кадров"),
            box_conf=st.slider("Порог рамки (detect)", 0.1, 0.9, p.box_conf, 0.05),
            ema_alpha=st.slider("Сглаживание (вес нового кадра)", 0.05, 1.0, p.ema_alpha, 0.05),
            confirm_frames=st.slider("Кадров для подтверждения", 1, 30, p.confirm_frames),
            clear_frames=st.slider("Кадров для отбоя", 1, 120, p.clear_frames),
        )
    if running:
        holder.monitor.update_params(params)

    with sb.expander("Telegram", expanded=False):
        if settings.telegram_configured:
            st.success("Бот настроен")
        else:
            st.warning("Не настроен. Укажите ECOWATCH_TG_TOKEN и ECOWATCH_TG_CHAT_ID в файле .env")
        notify = st.checkbox("Отправлять тревоги", True, disabled=not settings.telegram_configured)
        notify_photo = st.checkbox("Прикладывать снимок", True)
        notify_on_clear = st.checkbox("Сообщать об окончании угрозы", False)
        cooldown = st.slider("Пауза между тревогами, с", 0, 600, 60, 10)
        c1, c2 = st.columns(2)
        tg = TelegramNotifier(settings.telegram_token, settings.telegram_chat_id)
        if c1.button("Тест", disabled=not settings.telegram_configured, width="stretch"):
            ok = tg.send_message(f"✅ <b>EcoWatch AI</b> на связи\n{datetime.now():%d.%m.%Y %H:%M:%S}")
            (st.success if ok else st.error)("Отправлено" if ok else "Ошибка отправки")
        if c2.button("Отчёт", disabled=not settings.telegram_configured, width="stretch"):
            ok = tg.send_message(format_daily_report(store.day_summary()))
            (st.success if ok else st.error)("Отправлен" if ok else "Ошибка отправки")

    with sb.expander("Запись и карта", expanded=False):
        record = st.checkbox("Записывать клипы инцидентов", False)
        pre = st.slider("Секунд до тревоги", 0, 10, 3, disabled=not record)
        post = st.slider("Секунд после окончания", 2, 60, 10, disabled=not record)
        lat = st.number_input("Широта камеры", value=settings.camera_lat or 0.0, format="%.5f")
        lon = st.number_input("Долгота камеры", value=settings.camera_lon or 0.0, format="%.5f")
        has_geo = bool(lat or lon)

    sb.divider()
    if running:
        if sb.button("⏹ Остановить", type="primary", width="stretch"):
            holder.monitor.stop()
            st.rerun()
        return None, params
    if sb.button("▶ Запустить мониторинг", type="primary", width="stretch", disabled=spec is None):
        return MonitorOptions(
            source=spec,  # type: ignore[arg-type]
            params=params,
            notify=notify and settings.telegram_configured,
            notify_photo=notify_photo,
            notify_on_clear=notify_on_clear,
            notify_cooldown=float(cooldown),
            record=record,
            record_pre=float(pre),
            record_post=float(post),
            lat=lat if has_geo else None,
            lon=lon if has_geo else None,
        ), params
    return None, params


# ---------- вкладки ----------
@st.fragment(run_every=0.25)
def live_view(holder: MonitorHolder) -> None:
    mon = holder.monitor
    col_v, col_s = st.columns([3, 1.2])
    if mon is None:
        col_v.markdown("<div class='ew-idle'>📹&nbsp; Мониторинг не запущен</div>", unsafe_allow_html=True)
        col_s.markdown(status_card(None), unsafe_allow_html=True)
        return
    s = mon.state()
    if s.preview is not None:
        col_v.image(s.preview, width="stretch")
    else:
        col_v.markdown("<div class='ew-idle'>⏳ Подключение к источнику…</div>", unsafe_allow_html=True)
    if s.error:
        col_v.error(s.error)
    if s.finished:
        col_v.info("Видео обработано до конца.")
    elif not s.running:
        col_v.info("Мониторинг остановлен.")

    col_s.markdown(status_card(s.decision, f"Источник: {s.source}"), unsafe_allow_html=True)
    m1, m2 = col_s.columns(2)
    m1.metric("FPS", f"{s.fps:.1f}")
    m2.metric("Кадров", s.frames)
    m3, m4 = col_s.columns(2)
    m3.metric("Инцидентов", s.incidents)
    elapsed = (datetime.now() - s.started_at) if s.started_at else timedelta()
    m4.metric("Время", str(elapsed).split(".")[0])
    if s.recording:
        col_s.markdown("🔴 **Идёт запись клипа**")


def tab_image(detector: Detector, params: DetectionParams) -> None:
    st.caption("Проверка отдельного снимка — без сглаживания по времени.")
    up = st.file_uploader("Изображение", type=["jpg", "jpeg", "png", "webp", "bmp"], key="img_upload")
    if up is None:
        empty_state("🖼️", "Загрузите изображение для анализа")
        return
    frame = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Не удалось прочитать изображение")
        return
    result = detector.predict(frame, params.box_conf)
    d = evaluate_single(result, params)
    c1, c2 = st.columns([3, 1.2])
    c1.image(draw_overlay(frame, d, result.boxes, up.name), channels="BGR", width="stretch")
    c2.markdown(status_card(d), unsafe_allow_html=True)


def tab_analytics(df: pd.DataFrame) -> None:
    if df.empty:
        empty_state("📊", "Аналитика появится после первых инцидентов")
        return
    period = st.radio("Период", ["24 часа", "7 дней", "30 дней", "Всё время"], index=1, horizontal=True)
    days = {"24 часа": 1, "7 дней": 7, "30 дней": 30}.get(period)
    if days:
        df = df[df["started_at"] >= datetime.now() - timedelta(days=days)]
    if df.empty:
        empty_state("📊", "За выбранный период инцидентов нет")
        return

    cols = st.columns(4)
    cols[0].metric("Инцидентов", len(df))
    cols[1].metric("🔥 Пожаров", int((df["label"] == "fire").sum()))
    cols[2].metric("💨 Дыма", int((df["label"] == "smoke").sum()))
    cols[3].metric("Ср. длительность", f"{df['Длительность, с'].mean():.0f} с" if df["ended_at"].notna().any() else "—")

    colors = {f"{h.icon} {h.title}": h.color for h in HAZARDS.values()}
    layout = dict(margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.2))

    c1, c2 = st.columns(2)
    bucket = df["started_at"].dt.floor("h" if days == 1 else "D")
    by_time = df.assign(t=bucket).groupby(["t", "Тип"]).size().reset_index(name="n")
    by_time["t"] = by_time["t"].dt.strftime("%d.%m %H:00" if days == 1 else "%d.%m.%Y")
    fig = px.bar(
        by_time,
        x="t",
        y="n",
        color="Тип",
        color_discrete_map=colors,
        labels={"t": "", "n": ""},
        title="Инциденты по времени",
    )
    c1.plotly_chart(fig.update_layout(**layout), width="stretch")

    by_type = df["Тип"].value_counts().reset_index()
    by_type.columns = ["Тип", "n"]
    fig = px.pie(
        by_type, values="n", names="Тип", hole=0.5, color="Тип", color_discrete_map=colors, title="Распределение"
    )
    c2.plotly_chart(fig.update_layout(**layout), width="stretch")

    c3, c4 = st.columns(2)
    fig = px.histogram(
        df,
        x="Вероятность, %",
        color="Тип",
        nbins=20,
        color_discrete_map=colors,
        title="Пиковая вероятность",
        labels={"count": ""},
    )
    c3.plotly_chart(fig.update_layout(**layout, bargap=0.05), width="stretch")

    days_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    heat = np.zeros((7, 24), dtype=int)
    for ts in df["started_at"]:
        heat[ts.weekday(), ts.hour] += 1
    fig = go.Figure(
        go.Heatmap(
            z=heat,
            x=[f"{h}:00" for h in range(24)],
            y=days_ru,
            colorscale="YlOrRd",
            hovertemplate="%{y} %{x}<br>Инцидентов: %{z}<extra></extra>",
        )
    )
    c4.plotly_chart(
        fig.update_layout(
            title="Активность по часам", margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(autorange="reversed")
        ),
        width="stretch",
    )


def tab_map(df: pd.DataFrame) -> None:
    geo = df.dropna(subset=["lat", "lon"]) if not df.empty else df
    if geo.empty:
        empty_state(
            "🗺️", "Нет инцидентов с координатами. Укажите широту и долготу камеры в настройках «Запись и карта»."
        )
        return
    points = geo.groupby(["lat", "lon"]).agg(n=("id", "count"), last=("started_at", "max")).reset_index()
    fig = px.scatter_map(
        points,
        lat="lat",
        lon="lon",
        size="n",
        hover_data={"n": True, "last": True},
        zoom=11,
        height=520,
        color_discrete_sequence=["#ef4444"],
    )
    st.plotly_chart(fig.update_layout(margin=dict(t=0, b=0, l=0, r=0)), width="stretch")
    st.caption(f"Камер: {len(points)} · инцидентов: {int(points['n'].sum())}")


def tab_journal(df: pd.DataFrame, store: EventStore) -> None:
    if df.empty:
        empty_state("📋", "Журнал пуст")
        return
    c1, c2 = st.columns([2, 1])
    types = c1.multiselect("Тип", sorted(df["Тип"].unique()), default=sorted(df["Тип"].unique()))
    min_conf = c2.slider("Мин. вероятность, %", 0, 100, 0)
    view = df[df["Тип"].isin(types) & (df["Вероятность, %"] >= min_conf)]
    table = view[["id", "started_at", "ended_at", "Тип", "Вероятность, %", "Длительность, с", "source"]].rename(
        columns={"id": "№", "started_at": "Начало", "ended_at": "Конец", "source": "Источник"}
    )
    event = st.dataframe(
        table, width="stretch", height=360, hide_index=True, on_select="rerun", selection_mode="single-row"
    )

    rows = event.selection.rows if event and event.selection else []
    if rows:
        rec = view.iloc[rows[0]]
        st.markdown(f"#### Инцидент №{rec['id']} · {rec['Тип']}")
        a, b = st.columns(2)
        snapshot, clip = existing_file(rec["snapshot"]), existing_file(rec["clip"])
        if snapshot:
            a.image(str(snapshot), caption="Снимок в момент тревоги", width="stretch")
        else:
            a.caption("Снимок недоступен")
        if clip:
            b.video(str(clip))
            b.download_button("Скачать клип", clip.read_bytes(), clip.name)
        else:
            b.caption("Клип не записывался")
    else:
        st.caption("Выберите строку, чтобы посмотреть снимок и клип.")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇ CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        f"ecowatch_{datetime.now():%Y%m%d}.csv",
        "text/csv",
        width="stretch",
    )
    c2.download_button(
        "⬇ Отчёт за сегодня",
        format_daily_report(store.day_summary())
        .replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .encode("utf-8"),
        f"report_{datetime.now():%Y%m%d}.txt",
        width="stretch",
    )
    with c3.popover("🗑 Очистить журнал", width="stretch"):
        st.write("Удалить все записи? Снимки и клипы на диске останутся.")
        if st.button("Да, удалить", type="primary"):
            store.clear()
            st.rerun()


# ---------- точка входа ----------
def main() -> None:
    st.set_page_config(page_title="EcoWatch AI", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    settings = get_settings()
    try:
        detector = get_detector(str(settings.model_path), settings.device)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()
    store = get_store(str(settings.db_path))
    holder = get_holder()

    options, params = sidebar(settings, detector, holder, store)
    if options is not None:
        with holder.lock:
            if holder.monitor is None or not holder.monitor.alive:
                notifier = TelegramNotifier(settings.telegram_token, settings.telegram_chat_id)
                holder.monitor = Monitor(detector, store, settings, options, notifier, get_sender())
                holder.monitor.start()
        st.rerun()

    st.title("🛡️ EcoWatch AI")
    st.caption("Обнаружение пожаров и дыма в реальном времени")

    t_live, t_img, t_an, t_map, t_log = st.tabs(["Мониторинг", "Анализ снимка", "Аналитика", "Карта", "Журнал"])
    with t_live:
        live_view(holder)
    with t_img:
        tab_image(detector, params)
    df = incidents_df(store)
    with t_an:
        tab_analytics(df)
    with t_map:
        tab_map(df)
    with t_log:
        tab_journal(df, store)

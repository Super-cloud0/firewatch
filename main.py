import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import os
from mss import mss
import time
import json
from PIL import Image, ImageDraw, ImageFont
from collections import deque

st.set_page_config(page_title="EcoWatch AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "detection_history.json")
RECORD_DIR = os.path.join(SCRIPT_DIR, "recordings")
DANGER_CLASSES = {"fire", "trash", "smoke"}
MIN_CONF = 0.55
DETECT_CONF = 0.40
CLASSIFY_CONF = 0.72
CLASSIFY_GAP = 0.18
SMOOTH_WINDOW = 5

_defaults = {
    "detection_log": [], "session_start": datetime.now(), "total_frames": 0,
    "alert_count": 0, "last_tg_time": datetime.min, "tg_sent_count": 0,
    "video_writer": None, "record_start": 0,
    "label_buffer": deque(maxlen=SMOOTH_WINDOW),
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def load_history():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_to_history(entry):
    h = load_history()
    h.append(entry)
    h = h[-2000:]
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, default=str)
    except Exception:
        pass


def get_all_data():
    history = load_history()
    session_log = st.session_state.get("detection_log", [])
    seen = {json.dumps(e, default=str) for e in history}
    return history + [e for e in session_log if json.dumps(e, default=str) not in seen]


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0f1117; --surface:#1a1d26; --card:#21242f;
  --border:#2a2d3a; --text:#e2e4eb; --muted:#8b8fa3; --dim:#5c5f73;
  --green:#22c55e; --red:#ef4444; --amber:#eab308; --blue:#3b82f6; --violet:#a78bfa;
}
*{font-family:'Inter',system-ui,sans-serif!important}
.stApp{background:var(--bg)}

[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{font-size:.84rem}

h1{color:var(--text)!important;font-weight:700!important;font-size:1.5rem!important;letter-spacing:-.3px}
h2,h3,h4{color:var(--text)!important;font-weight:600!important}
.section-label{color:var(--dim);font-size:.65rem;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px}

.stTabs [data-baseweb="tab-list"]{gap:2px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:3px}
.stTabs [data-baseweb="tab"]{border-radius:6px;color:var(--muted);font-weight:600;font-size:.82rem;padding:7px 16px}
.stTabs [aria-selected="true"]{background:var(--card)!important;color:var(--text)!important;border-bottom:none!important}

.stButton>button{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:7px 16px;font-weight:600;font-size:.8rem;transition:.15s}
.stButton>button:hover{border-color:var(--green);color:var(--green)}
.stDownloadButton>button{background:var(--card)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:6px!important;font-weight:600!important}

[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;font-size:1.4rem!important}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-weight:500!important;font-size:.72rem!important}

.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin:4px 0}

.alert-danger{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-left:3px solid var(--red);color:var(--text);padding:14px 16px;border-radius:8px}
.alert-safe{background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.12);border-left:3px solid var(--green);color:var(--text);padding:14px 16px;border-radius:8px}

.video-idle{background:var(--surface);border:1px dashed var(--border);border-radius:8px;height:380px;display:flex;align-items:center;justify-content:center}

.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sg-item{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center}
.sg-val{font-size:1.3rem;font-weight:700;color:var(--text);line-height:1.1}
.sg-lbl{font-size:.62rem;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin-top:2px}

.sb-stats{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:10px;margin-top:10px;font-size:.76rem;color:var(--muted);line-height:1.7}

.app-footer{text-align:center;color:var(--dim);font-size:.7rem;padding:24px 0 8px;border-top:1px solid var(--border);margin-top:40px}

#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    paths = [
        (os.path.join(SCRIPT_DIR, "models", "best.pt"), "classify"),
        (os.path.join(SCRIPT_DIR, "shop project", "runs", "detect", "train", "weights", "best.pt"), "detect"),
        (os.path.join(SCRIPT_DIR, "shop project", "runs", "classify", "train_v2", "weights", "best.pt"), "classify"),
        (os.path.join(SCRIPT_DIR, "shop project", "runs", "classify", "train", "weights", "best.pt"), "classify"),
    ]
    for path, mtype in paths:
        if os.path.exists(path):
            return YOLO(path), mtype
    st.error("Модель не найдена. Поместите best.pt в папку models/")
    return None, None


model = None
MODEL_TYPE = None
try:
    model, MODEL_TYPE = load_model()
    if MODEL_TYPE == "classify":
        st.sidebar.warning("⚠️ Используется classify-модель. Для точного обнаружения обучите detect-модель (train.py)")
except Exception as e:
    st.error(f"Ошибка модели: {e}")

BOT_TOKEN = "8084419721:AAFRMEcaoORZwbCwLpnEKmmmtVK_bG-Xunw"
CHAT_ID = "1719237834"


def send_telegram_alert(message, image_array=None):
    import requests, io
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.json().get("ok"):
            return False
        if image_array is not None:
            ok, buf = cv2.imencode(".jpg", image_array)
            if ok:
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": f"Снимок {datetime.now():%d.%m.%Y %H:%M:%S}"},
                    files={"photo": ("alert.jpg", io.BytesIO(buf), "image/jpeg")},
                    timeout=10,
                )
        st.session_state["tg_sent_count"] += 1
        return True
    except Exception:
        return False


def send_daily_report():
    history = load_history()
    today = datetime.now().strftime("%Y-%m-%d")
    today_events = [e for e in history if e.get("time", "").startswith(today)]
    fires = sum(1 for e in today_events if e.get("label") == "fire")
    trash = sum(1 for e in today_events if e.get("label") == "trash")
    factory = sum(1 for e in today_events if e.get("label") == "factory")
    avg = sum(e.get("confidence", 0) for e in today_events) / max(len(today_events), 1)
    msg = (
        f"📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b>\n"
        f"📅 {datetime.now():%d.%m.%Y}\n\n"
        f"🔥 Пожаров: <b>{fires}</b>\n"
        f"🗑️ Сжигание мусора: <b>{trash}</b>\n"
        f"🏭 Заводской дым: <b>{factory}</b>\n"
        f"📈 Средняя вероятность: <b>{avg:.1f}%</b>\n"
        f"📋 Всего: <b>{len(today_events)}</b>\n\n"
        f"{'⚠️ Повышенная активность!' if fires + trash > 5 else '✅ Ситуация стабильна'}\n\n"
        f"<i>— EcoWatch AI</i>"
    )
    return send_telegram_alert(msg)


def process_frame(results, frame):
    r = results[0]

    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        names = r.names
        detections = []
        class_counts = {}
        best_conf = 0.0
        best_label = "normal"

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < DETECT_CONF:
                continue
            cls_id = int(boxes.cls[i])
            cls_name = names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            detections.append((x1, y1, x2, y2, conf, cls_name))

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            if conf > best_conf:
                best_conf = conf
                best_label = cls_name

        if not detections:
            return "normal", 0.0, {}, []

        total_det = sum(class_counts.values())
        scores = {name: class_counts.get(name, 0) / total_det for name in names.values()}
        return best_label, best_conf, scores, detections

    if hasattr(r, 'probs') and r.probs is not None:
        raw = r.probs.data.cpu().numpy().flatten()
        scores = {}
        for i, name in r.names.items():
            scores[name] = float(raw[i])

        final_label = max(scores, key=scores.get)
        final_conf = scores[final_label]

        if final_conf < CLASSIFY_CONF:
            return "normal", final_conf, scores, []
        sorted_vals = sorted(scores.values(), reverse=True)
        if len(sorted_vals) >= 2 and (sorted_vals[0] - sorted_vals[1]) < CLASSIFY_GAP:
            return "normal", final_conf, scores, []
        if final_label == "normal":
            return "normal", final_conf, scores, []

        return final_label, final_conf, scores, []

    return "normal", 0.0, {}, []


CLASS_COLORS = {
    "fire":    (60, 76, 231),
    "trash":   (11, 160, 232),
    "factory": (210, 130, 50),
    "smoke":   (180, 163, 148),
    "normal":  (80, 180, 80),
}

_pil_font_cache = {}


def _get_font(size=16):
    if size in _pil_font_cache:
        return _pil_font_cache[size]
    for name in ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        try:
            f = ImageFont.truetype(name, size)
            _pil_font_cache[size] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _pil_font_cache[size] = f
    return f


def put_cyrillic(img, text, pos, font_size=16, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_text_size_pil(text, font_size=16):
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def smooth_label(label, conf):
    buf = st.session_state.get("label_buffer", deque(maxlen=SMOOTH_WINDOW))
    buf.append(label)
    st.session_state["label_buffer"] = buf
    if label not in DANGER_CLASSES:
        return label, conf
    if len(buf) >= SMOOTH_WINDOW and all(l in DANGER_CLASSES for l in buf):
        return label, conf
    return "normal", conf


def draw_overlay(frame, label, confidence, scores, detections=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    color = CLASS_COLORS.get(label, (180, 180, 180))
    info = HAZARD_INFO.get(label, {})
    label_ru = info.get("ru", label.upper())

    if detections:
        for (x1, y1, x2, y2, det_conf, cls_name) in detections:
            box_color = CLASS_COLORS.get(cls_name, (180, 180, 180))
            det_info = HAZARD_INFO.get(cls_name, {})
            box_label = det_info.get("ru", cls_name.upper())

            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            text = f"{box_label} {det_conf*100:.0f}%"
            tw, th = get_text_size_pil(text, 14)
            cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 8, y1), box_color, -1)
            overlay = put_cyrillic(overlay, text, (x1 + 4, y1 - th - 6), font_size=14, color=(255, 255, 255))

            bw, bh = x2 - x1, y2 - y1
            cl = min(20, bw // 4, bh // 4)
            ct = 3
            cv2.line(overlay, (x1, y1), (x1 + cl, y1), box_color, ct)
            cv2.line(overlay, (x1, y1), (x1, y1 + cl), box_color, ct)
            cv2.line(overlay, (x2, y1), (x2 - cl, y1), box_color, ct)
            cv2.line(overlay, (x2, y1), (x2, y1 + cl), box_color, ct)
            cv2.line(overlay, (x1, y2), (x1 + cl, y2), box_color, ct)
            cv2.line(overlay, (x1, y2), (x1, y2 - cl), box_color, ct)
            cv2.line(overlay, (x2, y2), (x2 - cl, y2), box_color, ct)
            cv2.line(overlay, (x2, y2), (x2, y2 - cl), box_color, ct)

    bar_h = 36
    bar_overlay = overlay.copy()
    cv2.rectangle(bar_overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(bar_overlay, 0.65, overlay, 0.35, 0, overlay)

    dot_color = (0, 0, 255) if label in DANGER_CLASSES else (0, 200, 0)
    cv2.circle(overlay, (18, bar_h // 2), 5, dot_color, -1)

    status_text = f"  {label_ru}  |  {confidence*100:.0f}%  |  EcoWatch AI"
    overlay = put_cyrillic(overlay, status_text, (30, bar_h // 2 - 8), font_size=15, color=(220, 220, 220))

    time_text = datetime.now().strftime("%H:%M:%S")
    ttw, _ = get_text_size_pil(time_text, 14)
    overlay = put_cyrillic(overlay, time_text, (w - ttw - 12, bar_h // 2 - 7), font_size=14, color=(150, 150, 150))

    bottom_h = 28
    bot_y = h - bottom_h
    bot_overlay = overlay.copy()
    cv2.rectangle(bot_overlay, (0, bot_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bot_overlay, 0.6, overlay, 0.4, 0, overlay)

    bar_x = 10
    bar_w_max = w - 20
    bar_y = bot_y + 8
    bar_height = 12
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_height), (40, 40, 40), -1)
    fill_w = int(bar_w_max * confidence)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_height), color, -1)

    if not detections and label in DANGER_CLASSES and confidence > MIN_CONF:
        cv2.rectangle(overlay, (2, bar_h + 2), (w - 2, bot_y - 2), color, 3)

    return overlay


def build_probs_html(scores):
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name = sorted_items[0][0]
    lines = []
    for name, score in sorted_items:
        p = score * 100
        info = HAZARD_INFO.get(name, {})
        icon = info.get("icon", "")
        ru = info.get("ru", name)
        clr = info.get("color", "#8b8fa3")
        bar_w = max(int(p), 2)
        bold = "font-weight:700;" if name == top_name else "font-weight:400;opacity:.65;"
        lines.append(
            f"<div style='margin:3px 0;{bold}'>"
            f"<span style='color:{clr}'>{icon} {ru}</span>"
            f"<span style='float:right;color:#e2e4eb'>{p:.1f}%</span>"
            f"<div style='background:rgba(255,255,255,.06);border-radius:3px;height:4px;margin-top:2px'>"
            f"<div style='background:{clr};width:{bar_w}%;height:100%;border-radius:3px'></div></div></div>"
        )
    html = "\n".join(lines)

    if len(sorted_items) >= 2:
        gap = (sorted_items[0][1] - sorted_items[1][1]) * 100
        if gap < 12:
            html += (
                "<div style='margin-top:6px;padding:5px 8px;background:rgba(234,179,8,.08);"
                "border:1px solid rgba(234,179,8,.2);border-radius:4px;font-size:.7rem;color:#eab308'>"
                "Низкая разница — результат неоднозначен</div>"
            )

    return html


HAZARD_INFO = {
    "trash": {
        "ru": "СЖИГАНИЕ МУСОРА", "icon": "🗑️", "color": "#eab308",
        "desc": "Серый пятнистый дым · Хаотичная форма · Бытовые отходы",
        "tg_desc": "• Серый пятнистый дым\n• Хаотичная форма\n• Бытовые отходы, пластик",
    },
    "fire": {
        "ru": "ПОЖАР", "icon": "🔥", "color": "#ef4444",
        "desc": "Густой чёрный/белый дым · Вертикальный столб · Критическая опасность",
        "tg_desc": "• Густой чёрный/белый дым\n• Вертикальный столб\n• Критическая опасность",
    },
    "factory": {
        "ru": "ПРОИЗВОДСТВЕННЫЙ ДЫМ", "icon": "🏭", "color": "#3b82f6",
        "desc": "Белый однородный дым · Ровная струя · Промышленный выброс",
        "tg_desc": "• Белый однородный дым\n• Ровная струя\n• Промышленный выброс",
    },
    "smoke": {
        "ru": "ДЫМ", "icon": "💨", "color": "#94a3b8",
        "desc": "Обнаружен дым · Возможная угроза · Требуется внимание",
        "tg_desc": "• Обнаружен дым\n• Возможная угроза\n• Требуется внимание",
    },
    "normal": {
        "ru": "НОРМА", "icon": "✅", "color": "#22c55e",
        "desc": "Угроз не обнаружено · Чистый кадр",
        "tg_desc": "",
    },
}

with st.sidebar:
    logo_path = os.path.join(SCRIPT_DIR, "logo", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=140)
    else:
        st.markdown(
            "<div style='text-align:center;padding:6px 0 14px'>"
            "<div style='font-size:1.6rem'>🛡️</div>"
            "<div style='font-size:1rem;font-weight:700;color:#e2e4eb;margin-top:2px'>EcoWatch AI</div>"
            "<div style='font-size:.6rem;color:#5c5f73;margin-top:1px'>v3.1 · Smoke Detection</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-label">Источник</p>', unsafe_allow_html=True)
    source_type = st.selectbox(
        "Источник", ["Веб-камера", "IP-камера", "Загрузить видео", "Загрузить изображение", "Захват экрана"],
        label_visibility="collapsed",
    )

    if source_type == "Веб-камера":
        camera_id = st.number_input("ID камеры", 0, 10, 0)
    elif source_type == "IP-камера":
        ip_url = st.text_input("RTSP URL", "rtsp://")
    elif source_type == "Загрузить видео":
        uploaded_video = st.file_uploader("Видео", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    elif source_type == "Загрузить изображение":
        uploaded_image = st.file_uploader("Изображение", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-label">Telegram</p>', unsafe_allow_html=True)
    enable_telegram = st.checkbox("Отправлять алерты", value=True)
    send_photo_tg = st.checkbox("Прикреплять фото", value=True)
    tg_cooldown = st.slider("Интервал (сек)", 5, 60, 15)
    tg_threshold = st.slider("Мин. уверенность (%)", 30, 95, 50)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧪 Тест"):
            ok = send_telegram_alert(
                "✅ <b>Тест</b>\n\nEcoWatch AI работает.\n" + datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            )
            st.success("OK") if ok else st.error("Ошибка")
    with c2:
        if st.button("📊 Отчёт"):
            ok = send_daily_report()
            st.success("Отправлен") if ok else st.error("Ошибка")

    st.markdown("---")
    st.markdown('<p class="section-label">Дополнительно</p>', unsafe_allow_html=True)
    enable_sound = st.checkbox("Звуковой сигнал", value=False)
    enable_recording = st.checkbox("Запись при угрозе", value=False)
    if enable_recording:
        record_duration = st.slider("Длительность записи (сек)", 5, 60, 15)

    st.markdown(
        f"<div class='sb-stats'>"
        f"📨 TG: <b style='color:#3b82f6'>{st.session_state.get('tg_sent_count', 0)}</b> &nbsp;"
        f"🚨 Алертов: <b style='color:#ef4444'>{st.session_state.get('alert_count', 0)}</b><br>"
        f"🎞️ Кадров: <b style='color:#22c55e'>{st.session_state.get('total_frames', 0)}</b> &nbsp;"
        f"⏱️ Сессия: <b style='color:#a78bfa'>{str(datetime.now() - st.session_state['session_start']).split('.')[0]}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<h1>🛡️ EcoWatch AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#5c5f73;font-size:.82rem;margin:-8px 0 18px'>"
    "Система обнаружения экологических угроз</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Мониторинг", "Аналитика", "Карта", "Журнал"])

with tab1:
    col_v, col_s = st.columns([5, 2])
    with col_v:
        video_placeholder = st.empty()
        is_running = st.toggle("Запустить мониторинг", value=False, key="run_toggle")
        if not is_running:
            video_placeholder.markdown(
                "<div class='video-idle'>"
                "<div style='text-align:center'>"
                "<div style='font-size:2rem;opacity:.25'>📹</div>"
                "<p style='color:#5c5f73;font-size:.82rem;margin:10px 0 0'>Ожидание запуска</p>"
                "</div></div>",
                unsafe_allow_html=True,
            )
    with col_s:
        alert_placeholder = st.empty()
        stats_placeholder = st.empty()
        sound_placeholder = st.empty()
        if not is_running:
            alert_placeholder.markdown(
                "<div class='alert-safe'>"
                "<div style='font-weight:600;font-size:.85rem'>Система готова</div>"
                "<div style='color:#8b8fa3;font-size:.74rem;margin-top:3px'>"
                "Включите мониторинг для анализа</div></div>",
                unsafe_allow_html=True,
            )

    if is_running and model:
        cap = None
        sct = None
        frame_count = 0
        start_time = time.time()

        try:
            if source_type == "Веб-камера":
                cap = cv2.VideoCapture(int(camera_id))
            elif source_type == "IP-камера":
                cap = cv2.VideoCapture(ip_url)
            elif source_type == "Загрузить видео" and uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                cap = cv2.VideoCapture(tfile.name)
            elif source_type == "Захват экрана":
                sct = mss()
                monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
            elif source_type == "Загрузить изображение" and uploaded_image:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                results = model(frame)

                label, conf, scores, detections = process_frame(results, frame)
                info = HAZARD_INFO.get(label, {})
                label_ru = info.get("ru", label.upper())
                is_img_danger = label in DANGER_CLASSES and conf > MIN_CONF
                probs_html = build_probs_html(scores)

                annotated = draw_overlay(frame, label, conf, scores, detections)
                video_placeholder.image(annotated, channels="BGR")

                if is_img_danger:
                    alert_placeholder.markdown(
                        f"<div class='alert-danger'>"
                        f"<div style='font-weight:600;font-size:.9rem'>{info.get('icon', '')} {label_ru}</div>"
                        f"<div style='color:#fca5a5;font-size:.72rem;margin-top:3px'>{info.get('desc', '')}</div>"
                        f"<div style='margin-top:8px;font-size:.76rem'>{probs_html}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    entry = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": label_ru, "label": label,
                        "confidence": round(conf * 100, 1), "source": source_type,
                    }
                    st.session_state["detection_log"].append(entry)
                    save_to_history(entry)
                    st.session_state["alert_count"] += 1
                else:
                    alert_placeholder.markdown(
                        f"<div class='alert-safe'>"
                        f"<div style='font-weight:600;font-size:.85rem'>Угроз не обнаружено</div>"
                        f"<div style='margin-top:6px;font-size:.76rem'>{probs_html}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Ошибка: {e}")

        if cap or sct:
            while is_running:
                if sct:
                    img = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Конец потока")
                        break

                results = model(frame)
                label, conf, scores, detections = process_frame(results, frame)
                label, conf = smooth_label(label, conf)

                annotated = draw_overlay(frame, label, conf, scores, detections)
                video_placeholder.image(annotated, channels="BGR")

                frame_count += 1
                st.session_state["total_frames"] += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                info = HAZARD_INFO.get(label, {})
                label_ru = info.get("ru", label.upper())
                is_danger = label in DANGER_CLASSES and conf > MIN_CONF
                probs_html = build_probs_html(scores)

                if is_danger:
                    alert_placeholder.markdown(
                        f"<div class='alert-danger'>"
                        f"<div style='font-weight:600;font-size:.9rem'>{info.get('icon', '')} {label_ru}</div>"
                        f"<div style='color:#fca5a5;font-size:.72rem;margin-top:3px'>{info.get('desc', '')}</div>"
                        f"<div style='margin-top:8px;font-size:.76rem'>{probs_html}</div>"
                        f"<div style='color:#f87171;font-size:.68rem;margin-top:4px'>{datetime.now():%H:%M:%S}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if enable_sound:
                        sound_placeholder.markdown(
                            '<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg"></audio>',
                            unsafe_allow_html=True,
                        )
                else:
                    alert_placeholder.markdown(
                        f"<div class='alert-safe'>"
                        f"<div style='font-weight:600;font-size:.85rem'>Норма</div>"
                        f"<div style='margin-top:6px;font-size:.76rem'>{probs_html}</div>"
                        f"<div style='color:#8b8fa3;font-size:.68rem;margin-top:3px'>{datetime.now():%H:%M:%S}</div></div>",
                        unsafe_allow_html=True,
                    )
                    sound_placeholder.empty()

                dur = str(timedelta(seconds=int(elapsed)))
                stats_placeholder.markdown(
                    f"<div class='panel'><div class='sg'>"
                    f"<div class='sg-item'><div class='sg-val' style='color:#3b82f6'>{fps:.0f}</div><div class='sg-lbl'>FPS</div></div>"
                    f"<div class='sg-item'><div class='sg-val' style='color:#22c55e'>{frame_count}</div><div class='sg-lbl'>Кадров</div></div>"
                    f"<div class='sg-item'><div class='sg-val' style='color:#ef4444'>{st.session_state['alert_count']}</div><div class='sg-lbl'>Алертов</div></div>"
                    f"<div class='sg-item'><div class='sg-val' style='color:#a78bfa;font-size:.9rem'>{dur}</div><div class='sg-lbl'>Время</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                if is_danger and frame_count % 30 == 0:
                    entry = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": label_ru, "label": label,
                        "confidence": round(conf * 100, 1), "source": source_type,
                    }
                    st.session_state["detection_log"].append(entry)
                    save_to_history(entry)
                    st.session_state["alert_count"] += 1

                if enable_telegram and is_danger and conf > tg_threshold / 100.0:
                    diff = (datetime.now() - st.session_state.get("last_tg_time", datetime.min)).total_seconds()
                    if diff > tg_cooldown:
                        msg = (
                            f"🚨 <b>ЭКОЛОГИЧЕСКАЯ УГРОЗА</b>\n\n"
                            f"{info.get('icon', '')} <b>{label_ru}</b>\n"
                            f"📊 Вероятность: <b>{conf * 100:.1f}%</b>\n"
                            f"📹 Источник: {source_type}\n"
                            f"🕐 {datetime.now():%d.%m.%Y %H:%M:%S}\n\n"
                            f"{info.get('tg_desc', '')}\n\n"
                            f"<i>— EcoWatch AI</i>"
                        )
                        send_telegram_alert(msg, annotated if send_photo_tg else None)
                        st.session_state["last_tg_time"] = datetime.now()

                if enable_recording and is_danger:
                    if st.session_state.get("video_writer") is None:
                        os.makedirs(RECORD_DIR, exist_ok=True)
                        fname = os.path.join(RECORD_DIR, f"alert_{datetime.now():%Y%m%d_%H%M%S}.avi")
                        fh, fw = frame.shape[:2]
                        st.session_state["video_writer"] = cv2.VideoWriter(
                            fname, cv2.VideoWriter_fourcc(*"XVID"), 15, (fw, fh)
                        )
                        st.session_state["record_start"] = time.time()
                    wr = st.session_state.get("video_writer")
                    if wr:
                        wr.write(annotated)
                        if time.time() - st.session_state.get("record_start", 0) > record_duration:
                            wr.release()
                            st.session_state["video_writer"] = None

            if cap:
                cap.release()
            wr = st.session_state.get("video_writer")
            if wr:
                wr.release()
                st.session_state["video_writer"] = None


_PLOT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b8fa3", family="Inter"),
    margin=dict(t=30, b=20, l=20, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)", zerolinecolor="rgba(255,255,255,.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)", zerolinecolor="rgba(255,255,255,.04)"),
)

with tab2:
    all_data = get_all_data()
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        df["time"] = pd.to_datetime(df["time"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Всего", len(df))
        with m2:
            st.metric("Пожаров", len(df[df["label"] == "fire"]))
        with m3:
            st.metric("Мусор", len(df[df["label"] == "trash"]))
        with m4:
            st.metric("Заводской", len(df[df["label"] == "factory"]))

        c1, c2 = st.columns(2)
        with c1:
            daily = df.groupby(df["time"].dt.date).size().reset_index(name="count")
            fig = px.bar(daily, x="time", y="count", labels={"time": "", "count": ""}, color_discrete_sequence=["#22c55e"])
            fig.update_layout(**_PLOT, title="События по дням")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            tc = df["type"].value_counts().reset_index()
            tc.columns = ["Тип", "N"]
            fig = px.pie(tc, values="N", names="Тип", color_discrete_sequence=["#ef4444", "#eab308", "#3b82f6"], hole=0.45)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b8fa3", family="Inter"), margin=dict(t=30, b=20), title="Распределение",
            )
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.scatter(
                df.tail(100), x="time", y="confidence", color="type", size="confidence",
                labels={"time": "", "confidence": "%", "type": "Тип"},
                color_discrete_sequence=["#ef4444", "#eab308", "#3b82f6"],
            )
            fig.update_layout(**_PLOT, title="Вероятности обнаружений", showlegend=True, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            avg = df.groupby("type")["confidence"].mean().reset_index()
            avg.columns = ["Тип", "Ср. %"]
            fig = px.bar(avg, x="Тип", y="Ср. %", color="Тип", color_discrete_sequence=["#ef4444", "#eab308", "#3b82f6"])
            fig.update_layout(**_PLOT, title="Средняя уверенность", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if len(df) > 5:
            st.markdown("#### Активность по часам")
            df["hour"] = df["time"].dt.hour
            df["weekday"] = df["time"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
            hp = heat.pivot_table(index="weekday", columns="hour", values="count", fill_value=0)
            hp = hp.reindex([d for d in day_order if d in hp.index])
            day_ru = {"Monday": "Пн", "Tuesday": "Вт", "Wednesday": "Ср", "Thursday": "Чт", "Friday": "Пт", "Saturday": "Сб", "Sunday": "Вс"}
            fig = go.Figure(
                data=go.Heatmap(
                    z=hp.values,
                    x=[f"{hh}:00" for hh in hp.columns],
                    y=[day_ru.get(d, d) for d in hp.index],
                    colorscale=[[0, "#0f1117"], [0.5, "#22c55e"], [1, "#ef4444"]],
                    hovertemplate="%{y} %{x}<br>Событий: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b8fa3", family="Inter"), margin=dict(t=10, b=10, l=10, r=10), height=220,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            "<div class='panel' style='text-align:center;padding:50px 16px'>"
            "<div style='font-size:1.6rem;opacity:.25'>📊</div>"
            "<p style='color:#5c5f73;margin:10px 0 0'>Данные появятся после запуска мониторинга</p></div>",
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown("#### Карта обнаружений")
    st.caption("Координаты привязаны к расположению камер.")

    all_data_map = get_all_data()
    base_lat = st.number_input("Широта камеры", value=51.1694, format="%.4f", key="map_lat")
    base_lon = st.number_input("Долгота камеры", value=71.4491, format="%.4f", key="map_lon")

    if len(all_data_map) > 0:
        map_points = []
        for i, e in enumerate(all_data_map[-100:]):
            np.random.seed(hash(e.get("time", "")) % 2**31)
            map_points.append({"lat": base_lat + np.random.uniform(-0.015, 0.015), "lon": base_lon + np.random.uniform(-0.015, 0.015)})
        df_map = pd.DataFrame(map_points)
        st.map(df_map, latitude="lat", longitude="lon")
        st.markdown(
            f"<div class='panel' style='text-align:center'>"
            f"<span style='color:#8b8fa3;font-size:.8rem'>Показано <b style='color:#e2e4eb'>{len(map_points)}</b>"
            f" последних обнаружений</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='panel' style='text-align:center;padding:50px 16px'>"
            "<div style='font-size:1.6rem;opacity:.25'>🗺️</div>"
            "<p style='color:#5c5f73;margin:10px 0 0'>Нет данных</p></div>",
            unsafe_allow_html=True,
        )

with tab4:
    all_data_log = get_all_data()
    if len(all_data_log) > 0:
        df_l = pd.DataFrame(all_data_log)
        df_l["time"] = pd.to_datetime(df_l["time"])
        df_l = df_l.sort_values("time", ascending=False)

        fc1, fc2 = st.columns(2)
        with fc1:
            f_type = st.multiselect("Тип", df_l["type"].unique(), default=list(df_l["type"].unique()))
        with fc2:
            f_conf = st.slider("Мин. уверенность", 0, 100, 0, key="log_conf")

        filt = df_l[(df_l["type"].isin(f_type)) & (df_l["confidence"] >= f_conf)]
        disp = filt.rename(columns={"time": "Время", "type": "Тип", "confidence": "%", "source": "Источник"})
        cols = ["Время", "Тип", "%", "Источник"]
        disp = disp[[c for c in cols if c in disp.columns]]
        st.dataframe(disp, use_container_width=True, height=400)

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            csv = disp.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать CSV", csv, f"ecowatch_{datetime.now():%Y%m%d}.csv", "text/csv")
        with bc2:
            report = (
                f"ОТЧЁТ ECOWATCH AI\n{'=' * 40}\n"
                f"Дата: {datetime.now():%d.%m.%Y %H:%M}\n"
                f"Событий: {len(filt)}\n"
                f"Пожаров: {len(filt[filt['type'].str.contains('ПОЖАР', na=False)])}\n"
                f"Мусор: {len(filt[filt['type'].str.contains('МУСОР', na=False)])}\n"
                f"{'=' * 40}\n"
            )
            for _, r in filt.head(50).iterrows():
                report += f"\n{r.get('time', '')} | {r.get('type', '')} | {r.get('confidence', '')}%"
            st.download_button("Скачать отчёт", report.encode("utf-8"), f"report_{datetime.now():%Y%m%d}.txt")
        with bc3:
            if st.button("Очистить историю"):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                st.session_state["detection_log"] = []
                st.session_state["alert_count"] = 0
                st.rerun()
    else:
        st.markdown(
            "<div class='panel' style='text-align:center;padding:50px 16px'>"
            "<div style='font-size:1.6rem;opacity:.25'>📋</div>"
            "<p style='color:#5c5f73;margin:10px 0 0'>Журнал пуст</p></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='app-footer'>EcoWatch AI · 2025 · Infomatrix<br>YOLOv8 · Streamlit · OpenCV</div>",
    unsafe_allow_html=True,
)

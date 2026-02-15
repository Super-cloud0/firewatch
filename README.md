# EcoWatch AI

Система обнаружения экологических угроз на основе YOLOv8.

Анализирует видеопоток в реальном времени и определяет:
- 🔥 Пожар
- 💨 Дым
- 🗑️ Сжигание мусора

## Возможности

- Мониторинг с веб-камеры, IP-камеры, экрана или загруженного файла
- Уведомления в Telegram с фото
- Аналитика и графики
- Журнал обнаружений с экспортом в CSV
- Запись видео при обнаружении угрозы

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
streamlit run main.py
```

Или через батник:

```
run_app.bat
```

## Обучение модели

Подготовьте датасет в папке `dataset_detect/` со структурой:

```
dataset_detect/
├── train/
│   ├── fire/
│   ├── normal/
│   └── smoke/
└── valid/
    ├── fire/
    ├── normal/
    └── smoke/
```

Запустите:

```bash
python train.py
```

## Стек

- Python 3.11
- Streamlit
- YOLOv8 (ultralytics)
- OpenCV
- Plotly

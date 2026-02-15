import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_detect")
PROJECT_DIR = os.path.join(SCRIPT_DIR, "shop project", "runs", "classify")

BASE_MODEL = "yolov8s-cls.pt"
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 32
PATIENCE = 15


def check_dataset():
    train_dir = os.path.join(DATASET_DIR, "train")
    valid_dir = os.path.join(DATASET_DIR, "valid")

    if not os.path.isdir(train_dir):
        print(f"Папка train не найдена: {train_dir}")
        return False
    if not os.path.isdir(valid_dir):
        print(f"Папка valid не найдена: {valid_dir}")
        return False

    print("Датасет:")
    total = 0
    for split_name, split_dir in [("train", train_dir), ("valid", valid_dir)]:
        classes = sorted(os.listdir(split_dir))
        counts = {}
        for cls in classes:
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
                counts[cls] = n
                total += n
        parts = ", ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"  {split_name}: {parts}")

    print(f"  Всего: {total} изображений")
    print(f"  Классы: {classes}")
    return True


def train():
    if not check_dataset():
        return

    from ultralytics import YOLO

    print(f"Базовая модель: {BASE_MODEL}")
    print(f"Эпох: {EPOCHS}, Размер: {IMG_SIZE}px, Батч: {BATCH_SIZE}")
    print(f"Сохранение: {PROJECT_DIR}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATASET_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=PROJECT_DIR,
        name="train_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = os.path.join(PROJECT_DIR, "train_v2", "weights", "best.pt")
    models_dir = os.path.join(SCRIPT_DIR, "models")
    if os.path.exists(best_path):
        os.makedirs(models_dir, exist_ok=True)
        import shutil
        shutil.copy2(best_path, os.path.join(models_dir, "best.pt"))
        print(f"Модель обучена и скопирована в models/best.pt")
    else:
        print("Ошибка обучения. Проверь логи выше.")


if __name__ == "__main__":
    train()

"""Обучение модели EcoWatch AI.

Классификация (как текущая models/best.pt) — датасет из папок по классам:
    python train.py --data dataset_detect

Детекция с рамками — датасет в формате YOLO с data.yaml:
    python train.py --data fire_smoke/data.yaml --model yolov8s.pt --imgsz 640

Лучшие веса копируются в models/best.pt, прежняя модель сохраняется как models/best_<дата>.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def describe_classify_dataset(data: Path) -> bool:
    ok = True
    for split in ("train", "val", "valid"):
        d = data / split
        if not d.is_dir():
            continue
        counts = {
            c.name: sum(1 for f in c.iterdir() if f.suffix.lower() in IMAGE_EXT)
            for c in sorted(d.iterdir())
            if c.is_dir()
        }
        print(f"  {split}: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + f"  (всего {sum(counts.values())})")
        empty = [k for k, v in counts.items() if v == 0]
        if empty:
            print(f"  ⚠ пустые классы в {split}: {empty}")
            ok = False
    if not (data / "train").is_dir():
        print(f"Нет папки {data / 'train'}")
        return False
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "dataset_detect"), help="папка датасета или data.yaml")
    p.add_argument("--model", help="базовые веса (по умолчанию yolov8s-cls.pt / yolov8s.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, help="размер входа (224 для classify, 640 для detect)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", help="cpu, 0, 0,1 … (по умолчанию — автоматически)")
    p.add_argument("--name", default=f"run_{datetime.now():%Y%m%d_%H%M}")
    p.add_argument("--no-install", action="store_true", help="не копировать результат в models/best.pt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = Path(args.data)
    task = "detect" if data.suffix in {".yaml", ".yml"} else "classify"
    print(f"Задача: {task}, датасет: {data}")
    if not data.exists():
        print("Датасет не найден")
        return 1
    if task == "classify" and not describe_classify_dataset(data):
        return 1

    from ultralytics import YOLO

    base = args.model or ("yolov8s-cls.pt" if task == "classify" else "yolov8s.pt")
    imgsz = args.imgsz or (224 if task == "classify" else 640)
    model = YOLO(base)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(ROOT / "runs" / task),
        name=args.name,
        exist_ok=True,
        # Огонь и дым не бывают вверх ногами, а цвет важен — аугментации умеренные.
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        plots=True,
    )

    best = Path(model.trainer.best)
    if not best.is_file():
        print("Обучение не дало весов — смотрите лог выше")
        return 1
    print(f"Лучшие веса: {best}")
    if not args.no_install:
        target = ROOT / "models" / "best.pt"
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            backup = target.with_name(f"best_{datetime.now():%Y%m%d_%H%M%S}.pt")
            shutil.copy2(target, backup)
            print(f"Старая модель сохранена: {backup.name}")
        shutil.copy2(best, target)
        print(f"Модель установлена: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

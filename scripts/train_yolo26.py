"""
Phase 3 — Train or Fine-Tune YOLO26

Wrapper around Ultralytics training with deployment-friendly defaults.

Usage:
    python scripts/train_yolo26.py \
        --model yolo26n.pt \
        --data datasets/my_dataset/data.yaml \
        --imgsz 320 \
        --epochs 100 \
        --output models/pytorch/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(model_name: str, data: str, imgsz: int, epochs: int, output: str):
    model = YOLO(model_name)
    result = model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=1,
        device=None,
        plots=True,
    )

    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but {best} was not found")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_path)

    print(f"Best model copied to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune YOLO26")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="models/pytorch/best.pt")
    args = parser.parse_args()
    train(args.model, args.data, args.imgsz, args.epochs, args.output)


if __name__ == "__main__":
    main()

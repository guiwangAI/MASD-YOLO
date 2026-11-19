import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train MASD-YOLO.")
    parser.add_argument("--model", default="ultralytics/cfg/models/v8/masd-yolo.yaml", help="Model YAML path.")
    parser.add_argument("--data", default="dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="masd-yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(Path(args.model)))
    model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

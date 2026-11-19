import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a MASD-YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Checkpoint path, for example runs/train/masd-yolo/weights/best.pt.")
    parser.add_argument("--data", default="dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="masd-yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(Path(args.weights)))
    model.val(
        data=str(Path(args.data)),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

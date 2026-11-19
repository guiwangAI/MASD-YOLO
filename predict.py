import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run MASD-YOLO prediction.")
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Image, directory, video, or stream source.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="masd-yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(Path(args.weights)))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()


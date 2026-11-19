from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly split a YOLO images/labels dataset into train and val sets."
    )
    parser.add_argument("--src", required=True, type=Path, help="Dataset root containing images/ and labels/.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio. Default: 0.2.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Copy or move files to the output directory. Default: copy.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Create empty label files when an image has no matching label.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def split_items(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    if len(shuffled) > 1:
        val_count = max(1, min(len(shuffled) - 1, val_count))

    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_pair(
    image_path: Path,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    mode: str,
    allow_missing_labels: bool,
) -> None:
    rel_image = image_path.relative_to(images_dir)
    label_path = label_path_for(image_path, images_dir, labels_dir)
    rel_label = rel_image.with_suffix(".txt")

    transfer_file(image_path, out_dir / "images" / split / rel_image, mode)

    out_label = out_dir / "labels" / split / rel_label
    if label_path.exists():
        transfer_file(label_path, out_label, mode)
    elif allow_missing_labels:
        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_label.write_text("", encoding="utf-8")
    else:
        raise FileNotFoundError(f"Missing label for image: {image_path}")


def main() -> None:
    args = parse_args()
    src_dir = args.src.resolve()
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    out_dir = args.out.resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
    if out_dir == src_dir:
        raise ValueError("--out must be different from --src.")

    images = collect_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No image files found in: {images_dir}")

    train_images, val_images = split_items(images, args.val_ratio, args.seed)
    prepare_output(out_dir, args.overwrite)

    for image_path in train_images:
        write_pair(image_path, "train", images_dir, labels_dir, out_dir, args.mode, args.allow_missing_labels)
    for image_path in val_images:
        write_pair(image_path, "val", images_dir, labels_dir, out_dir, args.mode, args.allow_missing_labels)

    summary = {
        "source": str(src_dir),
        "output": str(out_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "mode": args.mode,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


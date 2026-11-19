"""Generate RSSVG-Rob"""

from __future__ import annotations

import argparse
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

AUG_PARAMS = {
    "gaussian_noise": {"std": 30},
    "salt_pepper": {"amount": 0.02},
    "motion_blur": {"kernel_size": 21, "angle": 45},
    "jpeg_artifact": {"quality": 10},
    "downscale": {"scale": 0.25},
    "brightness_high": {"beta": 70},
    "brightness_low": {"beta": -70},
    "low_contrast": {"alpha": 0.4},
    "shadow": {"intensity": 0.45},
    "cutout_small": {"n_holes": 5, "max_ratio": 0.10},
    "cutout_large": {"n_holes": 2, "max_ratio": 0.25},
}


def aug_gaussian_noise(img: np.ndarray, std: float = 30) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_salt_pepper(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    n = max(1, int(amount * h * w))

    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 255

    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 0
    return out


def aug_motion_blur(img: np.ndarray, kernel_size: int = 21, angle: float = 45) -> np.ndarray:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def aug_jpeg_artifact(img: np.ndarray, quality: int = 10) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    restored = np.array(Image.open(buffer).convert("RGB"))
    return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)


def aug_downscale(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    scale = float(np.clip(scale, 0.01, 1.0))
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_brightness(img: np.ndarray, beta: float = 70) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=float(beta))


def aug_low_contrast(img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=float(alpha), beta=0)


def aug_shadow(img: np.ndarray, intensity: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    intensity = float(np.clip(intensity, 0.0, 1.0))

    x_grad = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y_grad = np.linspace(0, 0.35, h, dtype=np.float32)[:, None]
    offset = random.uniform(-0.25, 0.25)
    band = (x_grad + y_grad + offset) % 1.0

    mask = np.ones((h, w), dtype=np.float32)
    mask[band < 0.35] = 1.0 - intensity
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(h, w) * 0.03)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def aug_cutout(img: np.ndarray, n_holes: int = 5, max_ratio: float = 0.10) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    max_ratio = float(np.clip(max_ratio, 0.01, 1.0))
    max_hole_w = max(1, int(w * max_ratio))
    max_hole_h = max(1, int(h * max_ratio))

    for _ in range(max(1, int(n_holes))):
        hole_w = random.randint(1, max_hole_w)
        hole_h = random.randint(1, max_hole_h)
        x1 = random.randint(0, max(0, w - hole_w))
        y1 = random.randint(0, max(0, h - hole_h))
        fill = int(np.random.randint(0, 256))
        out[y1 : y1 + hole_h, x1 : x1 + hole_w] = fill

    return out


AUGMENTATIONS: dict[str, Callable[..., np.ndarray]] = {
    "gaussian_noise": aug_gaussian_noise,
    "salt_pepper": aug_salt_pepper,
    "motion_blur": aug_motion_blur,
    "jpeg_artifact": aug_jpeg_artifact,
    "downscale": aug_downscale,
    "brightness_high": aug_brightness,
    "brightness_low": aug_brightness,
    "low_contrast": aug_low_contrast,
    "shadow": aug_shadow,
    "cutout_small": aug_cutout,
    "cutout_large": aug_cutout,
}


def normalize_exts(exts: list[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}


def collect_images(images_dir: Path, image_exts: set[str]) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    )


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def load_dataset_metadata(src_root: Path) -> dict:
    yaml_path = src_root / "data.yaml"
    if not yaml_path.exists():
        return {"nc": 1, "names": ["ship"]}

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return {
        "nc": data.get("nc", 1),
        "names": data.get("names", ["ship"]),
    }


def write_dataset_yaml(src_root: Path, out_root: Path) -> None:
    metadata = load_dataset_metadata(src_root)
    data = {
        "path": str(out_root.resolve()),
        "train": "images",
        "val": "images",
        "nc": metadata["nc"],
        "names": metadata["names"],
    }

    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def copy_label(src_labels_dir: Path, dst_labels_dir: Path, image_rel: Path) -> None:
    src_label = src_labels_dir / image_rel.with_suffix(".txt")
    dst_label = dst_labels_dir / image_rel.with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.touch()


def process_augmentation(
    name: str,
    src_root: Path,
    dst_root: Path,
    images: list[Path],
    overwrite: bool,
) -> None:
    src_images_dir = src_root / "images"
    src_labels_dir = src_root / "labels"
    out_root = dst_root / name
    out_images_dir = out_root / "images"
    out_labels_dir = out_root / "labels"

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_root}")
        shutil.rmtree(out_root)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    fn = AUGMENTATIONS[name]
    params = AUG_PARAMS[name]

    for image_path in tqdm(images, desc=name, unit="image"):
        image_rel = image_path.relative_to(src_images_dir)
        image = read_image(image_path)
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        augmented = fn(image, **params)
        write_image(out_images_dir / image_rel, augmented)
        copy_label(src_labels_dir, out_labels_dir, image_rel)

    write_dataset_yaml(src_root, out_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate robustness evaluation datasets for YOLO-format data."
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Dataset root with images/ and labels/.")
    parser.add_argument("--dst-root", type=Path, required=True, help="Output root.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(AUGMENTATIONS),
        help="Run only selected augmentations.",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=sorted(DEFAULT_IMAGE_EXTS),
        help="Image suffixes to include.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    src_images_dir = src_root / "images"
    src_labels_dir = src_root / "labels"

    if not src_images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {src_images_dir}")
    if not src_labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {src_labels_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_exts = normalize_exts(args.image_exts)
    images = collect_images(src_images_dir, image_exts)
    if not images:
        raise RuntimeError(f"No images found in: {src_images_dir}")

    dst_root.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(AUGMENTATIONS)

    print(f"Found {len(images)} images.")
    for name in selected:
        process_augmentation(name, src_root, dst_root, images, args.overwrite)
    print(f"Done: {dst_root}")


if __name__ == "__main__":
    main()

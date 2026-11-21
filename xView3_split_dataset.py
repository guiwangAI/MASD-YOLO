import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from typing import NamedTuple

import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.strtree import STRtree


SIZE = 1024
STRIDE = 768
VAL_RATIO = 0.2
SEED = 42
SCENE_IDS = (
    "05bc615a9b0e1159t",
    "2899cfb18883251bt",
    "590dd08f71056cacv",
    "72dba3e82f782f67t",
    "b1844cde847a3942v",
    "cbe4ad26fe73f118t",
    "e98ca5aba8849b06t",
)


class Patch(NamedTuple):
    scene: str
    row: int
    col: int


class Scene(NamedTuple):
    path: Path
    height: int
    width: int
    transform: rasterio.Affine
    projector: Transformer


def read_scenes(raw):
    scenes = {}
    for scene in SCENE_IDS:
        path = raw / scene / "VH_dB.tif"
        if not path.is_file():
            subset = "train" if scene.endswith("t") else "validation"
            path = raw / subset / scene / "VH_dB.tif"
        with rasterio.open(path) as src:
            projector = Transformer.from_crs(src.crs, "EPSG:32631", always_xy=True)
            scenes[scene] = Scene(path, src.height, src.width, src.transform, projector)
    return scenes


def window_starts(center, extent):
    first = max(0, int((center - SIZE) // STRIDE) * STRIDE)
    last = min(extent - SIZE, int(center))
    return [start for start in range(first, last + 1, STRIDE) if start <= center < start + SIZE]


def read_patch_labels(raw, scenes):
    labels = defaultdict(list)
    for filename in ("train.csv", "validation.csv"):
        with (raw / filename).open(newline="", encoding="utf-8-sig") as stream:
            for item in csv.DictReader(stream):
                scene_id = item["scene_id"]
                if scene_id not in scenes or item["is_vessel"].lower() != "true":
                    continue
                scene = scenes[scene_id]
                row = float(item["detect_scene_row"])
                col = float(item["detect_scene_column"])
                length = float(item["vessel_length_m"] or "nan")
                side = (length / 10.0 if math.isfinite(length) and length > 0 else 20.0) / SIZE
                for top in window_starts(row, scene.height):
                    for left in window_starts(col, scene.width):
                        x, y = (col - left) / SIZE, (row - top) / SIZE
                        labels[Patch(scene_id, top, left)].append(
                            f"0 {x:.6f} {y:.6f} {side:.6f} {side:.6f}"
                        )
    return labels


def overlap_groups(patches, scenes):
    polygons = []
    for scene_id, y, x in patches:
        scene = scenes[scene_id]
        corners = ((x, y), (x + SIZE, y), (x + SIZE, y + SIZE), (x, y + SIZE))
        polygons.append(Polygon([
            scene.projector.transform(*(scene.transform * point)) for point in corners
        ]))
    tree = STRtree(polygons)
    neighbors = [set() for _ in patches]
    for i, polygon in enumerate(polygons):
        for j in tree.query(polygon, predicate="intersects"):
            if j > i and polygon.intersection(polygons[j]).area > 1e-4:
                neighbors[i].add(j)
                neighbors[j].add(i)
    groups, seen = [], set()
    for i in range(len(patches)):
        if i in seen:
            continue
        group = [i]
        seen.add(i)
        for j in group:
            unseen = neighbors[j] - seen
            seen.update(unseen)
            group.extend(unseen)
        groups.append(sorted(group))
    return groups


def split_patches(labels, scenes):
    patches = sorted(labels, key=lambda p: f"{p.scene}_{p.row}_{p.col}")
    target = int(len(patches) * VAL_RATIO)
    groups = overlap_groups(patches, scenes)
    random.Random(SEED).shuffle(groups)
    choices = {0: []}
    for group in groups:
        for count, indices in list(choices.items()):
            total = count + len(group)
            if total <= target and total not in choices:
                choices[total] = indices + group
    if target == 0 or target not in choices:
        raise ValueError(f"Cannot allocate {target} validation patches using whole overlap groups.")
    validation_indices = set(choices[target])
    train = [patch for i, patch in enumerate(patches) if i not in validation_indices]
    val = [patch for i, patch in enumerate(patches) if i in validation_indices]
    return {"train": train, "val": val}


def write_dataset(output, splits, scenes, labels):
    rng = random.Random(SEED)
    windows = defaultdict(lambda: defaultdict(list))
    for split, patches in splits.items():
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir()
        rng.shuffle(patches)
        for index, patch in enumerate(patches, 1):
            filename = f"{split}_{index:06d}"
            windows[patch.scene][patch.row].append((patch, split, filename))

    with rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024):
        for scene, rows in windows.items():
            with rasterio.open(scenes[scene].path) as src:
                for row, patches in sorted(rows.items()):
                    left = min(patch.col for patch, _, _ in patches)
                    right = max(patch.col + SIZE for patch, _, _ in patches)
                    stripe = src.read(1, window=Window(left, row, right - left, SIZE))
                    for patch, split, filename in patches:
                        col = patch.col - left
                        values = stripe[:, col:col + SIZE].astype(np.float32)
                        valid = np.isfinite(values) & (values != src.nodata)
                        image = np.zeros((SIZE, SIZE), dtype=np.uint8)
                        image[valid] = ((np.clip(values[valid], -40, 0) + 40) * (255 / 40)).astype(np.uint8)
                        Image.fromarray(image).save(output / split / "images" / f"{filename}.png", compress_level=3)
                        label = "\r\n".join(labels[patch]).encode("ascii")
                        (output / split / "labels" / f"{filename}.txt").write_bytes(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")

    scenes = read_scenes(args.raw)
    labels = read_patch_labels(args.raw, scenes)
    splits = split_patches(labels, scenes)
    with TemporaryDirectory(prefix=".fig11-build-", dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        write_dataset(staged, splits, scenes, labels)
        (staged / "data.yaml").write_text(
            f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
            "train: train/images\nval: val/images\nnames:\n  0: vessel\n",
            encoding="utf-8",
        )
        staged.rename(output)
    print(f"Train: {len(splits['train'])}, validation: {len(splits['val'])}")
    print(output)


if __name__ == "__main__":
    main()

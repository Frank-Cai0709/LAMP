"""Recover ISIC source filenames using exact segmentation-mask matching."""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


DATASETS = {
    "isic2017": {
        "root": "dataset_isic17",
        "image_dir": "ISIC2017_Task1-2_Training_Input",
        "mask_dir": "ISIC2017_Task1_Training_GroundTruth",
        "binary_mask": False,
    },
    "isic2018": {
        "root": "dataset_isic18",
        "image_dir": "ISIC2018_Task1-2_Training_Input",
        "mask_dir": "ISIC2018_Task1_Training_GroundTruth",
        "binary_mask": True,
    },
}


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype=np.uint8).tobytes()).hexdigest()


def prepared_image(image_path):
    image = imageio.imread(image_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def prepared_mask(mask_path, binary_mask):
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    if binary_mask:
        mask = mask > 0
    return mask


def recover(dataset, data_root, output):
    spec = DATASETS[dataset]
    root = data_root / spec["root"]
    image_dir = root / spec["image_dir"]
    mask_dir = root / spec["mask_dir"]
    source_by_digest = {}

    mask_paths = sorted(mask_dir.glob("ISIC_*_segmentation.png"))
    for mask_path in mask_paths:
        mask = prepared_mask(mask_path, spec["binary_mask"])
        key = digest(mask)
        if key in source_by_digest:
            other = source_by_digest[key]
            raise RuntimeError(
                f"Non-unique source mask: {other.name} and {mask_path.name}")
        source_by_digest[key] = mask_path

    rows = []
    used = set()
    split_lists = {}
    split_names = {"train": "train", "val": "validation", "test": "test"}
    for split in ("train", "val", "test"):
        images = np.load(root / f"data_{split}.npy", mmap_mode="r")
        masks = np.load(root / f"mask_{split}.npy", mmap_mode="r")
        names = []
        for index, (image, mask) in enumerate(zip(images, masks)):
            key = digest(mask)
            if key not in source_by_digest:
                raise RuntimeError(f"No exact mask match: {dataset} {split}[{index}]")
            mask_path = source_by_digest[key]
            image_path = image_dir / mask_path.name.replace("_segmentation.png", ".jpg")
            if image_path.name in used:
                raise RuntimeError(f"Source matched more than once: {image_path.name}")
            source_image = prepared_image(image_path).astype(np.float64)
            target_image = np.asarray(image, dtype=np.float64)
            image_mae = float(np.mean(np.abs(source_image - target_image)))
            image_correlation = float(np.corrcoef(
                source_image.reshape(-1), target_image.reshape(-1))[0, 1])
            if image_mae > 3.0 or image_correlation < 0.99:
                raise RuntimeError(
                    f"Image verification failed: {dataset} {split}[{index}] "
                    f"{image_path.name}, MAE={image_mae}, r={image_correlation}")
            used.add(image_path.name)
            names.append(image_path.name)
            rows.append({
                "split": split_names[split],
                "npy_index": index,
                "image_filename": image_path.name,
                "mask_filename": mask_path.name,
                "mask_sha256": key,
                "image_mae": f"{image_mae:.9f}",
                "image_correlation": f"{image_correlation:.9f}",
            })
        split_lists[split] = names

    if len(used) != len(mask_paths):
        missing = sorted(
            path.name.replace("_segmentation.png", ".jpg")
            for path in mask_paths
            if path.name.replace("_segmentation.png", ".jpg") not in used)
        raise RuntimeError(f"Unmatched source files ({len(missing)}): {missing[:10]}")

    for split, names in split_lists.items():
        (output / f"{dataset}_{split}.txt").write_text(
            "".join(f"{name}\n" for name in names), encoding="utf-8")

    csv_path = output / f"{dataset}_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recovered {len(rows)} exact mask matches for {dataset}: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset or DATASETS:
        recover(dataset, args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

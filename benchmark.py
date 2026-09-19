import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.losses import BCEDiceLoss
from utils.utils import load_checkpoint, save_json, set_seed


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unavailable"


def total_ram_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def environment_record():
    return {
        "os_platform": platform.platform(),
        "cpu_model": cpu_model(),
        "ram_bytes": total_ram_bytes(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_model": torch.cuda.get_device_name(torch.cuda.current_device()),
    }


def save_composite(image, mask, prediction, path, compress_level):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    mask = (mask[0].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
    prediction = (prediction[0].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
    mask = np.repeat(mask[..., None], 3, axis=2)
    prediction = np.repeat(prediction[..., None], 3, axis=2)
    Image.fromarray(np.concatenate((image, mask, prediction), axis=0)).save(
        path, format="PNG", compress_level=compress_level)


def measure_end_to_end(model, loader, output_dir, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = BCEDiceLoss()
    image_count, losses = 0, []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for images, masks, indices in loader:
            images = images.cuda(non_blocking=True).float()
            masks = masks.cuda(non_blocking=True).float()
            predictions = model(images)
            losses.append(criterion(predictions, masks).item())
            if settings["save_composite_png"]:
                for image, mask, prediction, index in zip(images, masks, predictions, indices):
                    save_composite(image, mask, prediction,
                                   output_dir / f"{int(index):04d}.png",
                                   settings["png_compress_level"])
            image_count += images.size(0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "images": image_count,
        "total_seconds": elapsed,
        "latency_seconds_per_image": elapsed / image_count,
        "throughput_images_per_second": image_count / elapsed,
        "mean_loss": float(np.mean(losses)),
        "includes": ["data_loading", "host_to_device_transfer", "forward",
                     "loss", "postprocessing", "png_writing"],
    }


def main():
    parser = argparse.ArgumentParser(description="Measure LAMP inference speed.")
    parser.add_argument("--config", default="configs/lamp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/benchmark")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the reported timing protocol.")
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    settings = cfg["benchmark"]
    model = build_model(cfg).cuda().eval()
    load_checkpoint(model, args.checkpoint, "cuda")
    dataset = build_dataset(cfg["dataset"], "test", False)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False,
                        num_workers=settings["num_workers"], pin_memory=True)
    output = Path(args.output)
    end_to_end = measure_end_to_end(model, loader, output / "composites", settings)
    result = {
        "environment": environment_record(),
        "checkpoint": str(Path(args.checkpoint)),
        "image_size": cfg["dataset"]["image_size"],
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        "timing_mode": "end_to_end_per_image",
        "number_of_evaluated_images": end_to_end["images"],
        "end_to_end": end_to_end,
    }
    save_json(result, output / "benchmark.json")
    print(f"End-to-end: {end_to_end['latency_seconds_per_image'] * 1000:.3f} ms/image, "
          f"{end_to_end['throughput_images_per_second']:.2f} images/s")
    print(f"Saved: {output / 'benchmark.json'}")


if __name__ == "__main__":
    main()

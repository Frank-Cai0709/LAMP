#!/usr/bin/env python3
"""Measure Table 18/24/25 quantities from a checkpoint and test split."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.utils import load_checkpoint


def boundary(mask):
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = -F.max_pool2d(-mask, 3, 1, 1)
    return (dilated != eroded).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-fraction", type=float, default=.05)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    loader = DataLoader(build_dataset(config["dataset"], args.split, False),
                        batch_size=1, shuffle=False, num_workers=0)
    records = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            image, mask = batch[:2]
            image, mask = image.to(device), mask.to(device).float()
            prediction, maps = model(image, return_maps=True)
            error_boundary = ((prediction >= .5) != (mask >= .5)).float() * boundary(mask)
            response = F.interpolate(maps["stage3"], mask.shape[-2:], mode="bilinear",
                                     align_corners=True).flatten()
            target = error_boundary.flatten()
            count = max(1, round(response.numel() * args.top_fraction))
            top = torch.topk(response, count).indices
            concentration = target[top].sum() / target.sum().clamp_min(1)
            row = {"index": index, "top5_error_concentration": float(concentration)}
            for stage in (1, 2, 3):
                module = getattr(model, f"c3f{stage}")
                discrepancy = getattr(module, "last_feature_discrepancy", None)
                correction = getattr(module, "last_correction", None)
                if discrepancy is not None:
                    row[f"stage{stage}_feature_mse"] = float(discrepancy.square().mean())
                if correction is not None:
                    u = maps[f"stage{stage}"]
                    magnitude = correction.abs().mean(1, keepdim=True)
                    median = u.median()
                    row[f"stage{stage}_correction_low"] = float(magnitude[u <= median].mean())
                    row[f"stage{stage}_correction_high"] = float(magnitude[u > median].mean())
            records.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in records for k in r}))
        writer.writeheader(); writer.writerows(records)
    summary = {key: {"mean": float(np.mean(values)), "std": float(np.std(values))}
               for key in records[0] if key != "index"
               for values in [[row[key] for row in records if key in row]]}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

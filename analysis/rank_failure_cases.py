#!/usr/bin/env python3
"""Rank test images by per-image DSC for auditable failure-case selection."""
import argparse, csv, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, type=Path); p.add_argument("--split", default="test")
    a = p.parse_args(); cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval(); load_checkpoint(model, a.checkpoint, device)
    loader = DataLoader(build_dataset(cfg["dataset"], a.split, False), batch_size=1,
                        shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for images, masks, indices in loader:
            pred = model(images.to(device)) >= cfg["evaluation"]["threshold"]
            target = masks.to(device) >= .5
            inter = (pred & target).sum().item(); total = pred.sum().item() + target.sum().item()
            rows.append({"index": int(indices[0]), "dsc": 1.0 if total == 0 else 2 * inter / total})
    rows.sort(key=lambda row: row["dsc"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("rank", "index", "dsc")); writer.writeheader()
        writer.writerows({"rank": rank, **row} for rank, row in enumerate(rows, 1))


if __name__ == "__main__": main()

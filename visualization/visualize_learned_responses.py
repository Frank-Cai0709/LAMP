#!/usr/bin/env python3
"""Create the five-column learned-response panel used by Figure 11."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.utils import load_checkpoint
from visualization.common import save_panel


def overlay(image, heatmap):
    import matplotlib.pyplot as plt
    image = image / max(float(image.max()), 1e-8)
    color = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip(.55 * image + .45 * color, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--case-ids", type=Path,
                        default=Path(__file__).with_name("case_ids.json"))
    parser.add_argument("--figure", default="Figure_11_GradCAM")
    parser.add_argument("--case-position", type=int,
                        help="Zero-based entry in the selected case list")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.index is None:
        if args.case_position is None:
            parser.error("provide --index or --case-position")
        cases = json.loads(args.case_ids.read_text(encoding="utf-8"))[args.figure]
        case = cases[args.case_position]
        configured = cfg["dataset"]["name"].replace("_", "").lower()
        recorded = case["dataset"].replace("_", "").lower()
        if configured != recorded:
            parser.error(f"selected case is {case['dataset']}, but config uses "
                         f"{cfg['dataset']['name']}")
        args.index = int(case["index"])
        args.split = case.get("split", args.split)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    image, mask, _ = build_dataset(cfg["dataset"], args.split, False)[args.index]
    activation = {}
    def hook(_module, _inputs, output):
        activation["value"] = output
        output.retain_grad()
    handle = model.encoder6.register_forward_hook(hook)
    prediction, maps = model(image.unsqueeze(0).to(device), return_maps=True)
    prediction.mean().backward()
    handle.remove()
    feature = activation["value"]
    weights = feature.grad.mean(dim=(-2, -1), keepdim=True)
    cam = F.relu((weights * feature).sum(1, keepdim=True))
    cam = F.interpolate(cam, mask.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    uncertainty = F.interpolate(maps["stage3"], mask.shape[-2:], mode="bilinear",
                                align_corners=True)[0, 0]
    raw = image.permute(1, 2, 0).cpu().numpy()
    raw = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-8)
    save_panel([raw, mask[0].numpy(), prediction.detach()[0, 0].cpu().numpy(),
                overlay(raw, uncertainty.detach().cpu().numpy()),
                overlay(raw, cam.detach().cpu().numpy())],
               ["Input", "Ground Truth", "Prediction", "Unreliability", "Grad-CAM"],
               args.output)


if __name__ == "__main__":
    main()

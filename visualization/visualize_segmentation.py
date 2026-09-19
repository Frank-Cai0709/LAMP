import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model
from utils.config import load_config
from utils.utils import load_checkpoint
from visualization.common import load_image, save_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lamp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/segmentation.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    image, tensor = load_image(args.image, cfg["dataset"]["image_size"], device)
    with torch.no_grad():
        prediction = model(tensor)[0, 0].cpu().numpy()
    save_panel([image, prediction, prediction >= cfg["evaluation"]["threshold"]],
               ["Input", "Prediction", "Binary mask"], args.output, "gray")


if __name__ == "__main__":
    main()

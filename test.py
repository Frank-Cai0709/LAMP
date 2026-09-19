import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.config import load_config
from utils.metrics import add_counts, confusion_counts, metrics_from_counts
from utils.utils import load_checkpoint, save_json, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LAMP variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    dataset = build_dataset(cfg["dataset"], "test", False)
    loader = DataLoader(dataset, batch_size=cfg["evaluation"]["batch_size"],
                        shuffle=False, num_workers=cfg["dataset"]["num_workers"])
    output = Path(args.output or cfg["experiment"]["output_dir"])
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    model.eval()
    with torch.no_grad():
        for images, masks, indices in loader:
            predictions = model(images.to(device)).cpu().numpy()
            add_counts(counts, confusion_counts(predictions, masks.numpy(),
                                                cfg["evaluation"]["threshold"]))
            if cfg["evaluation"]["save_predictions"]:
                for prediction, index in zip(predictions, indices):
                    image = (np.clip(prediction[0], 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(image).save(prediction_dir / f"{int(index):04d}.png")
    metrics = metrics_from_counts(counts)
    save_json(metrics, output / "metrics.json")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

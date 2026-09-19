#!/usr/bin/env python3
"""Plot train/validation epoch losses recorded by train.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="LAMP loss")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    epochs = [row["epoch"] for row in rows]
    train = [row["loss"] for row in rows]
    val = [row["val_loss"] for row in rows]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train, label="Training Loss", color="#629e6e")
    plt.plot(epochs, val, label="Validation Loss", color="#1a4673")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(alpha=.3)
    plt.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()

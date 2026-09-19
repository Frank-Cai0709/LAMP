import argparse
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualization.common import save_panel


def main():
    parser = argparse.ArgumentParser(description="Compare aligned prediction images.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="results/ablation.png")
    args = parser.parse_args()
    if len(args.images) != len(args.labels):
        raise ValueError("--images and --labels must have the same length")
    save_panel([Image.open(path) for path in args.images], args.labels, args.output, "gray")


if __name__ == "__main__":
    main()

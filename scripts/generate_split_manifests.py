"""Generate readable split text files from explicit filename metadata."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True,
                        help="JSON containing train, validation, and test filename arrays.")
    parser.add_argument("--dataset", required=True, help="Output filename prefix.")
    parser.add_argument("--output", default="splits")
    args = parser.parse_args()
    with Path(args.metadata).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mapping = {"train": "train", "validation": "val", "test": "test"}
    for source_key, suffix in mapping.items():
        names = metadata.get(source_key)
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Missing filename list: {source_key}")
        destination = output / f"{args.dataset}_{suffix}.txt"
        destination.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
        print(f"Saved {len(names)} filenames to {destination}")


if __name__ == "__main__":
    main()

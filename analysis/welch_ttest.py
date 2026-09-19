"""Compute two-sided Welch tests from saved per-seed scalar metrics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import ttest_ind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with method, seed, and metric columns.")
    parser.add_argument("--reference", required=True, help="Reference method name.")
    parser.add_argument("--metric", default="DSC")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    values = defaultdict(list)
    seeds = defaultdict(list)
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["method"]].append(float(row[args.metric]))
            seeds[row["method"]].append(row["seed"])
    if args.reference not in values:
        raise ValueError(f"Reference method not found: {args.reference}")
    if len(values[args.reference]) < 2:
        raise ValueError("Welch's test requires at least two reference runs.")

    comparisons = []
    for method in sorted(values):
        if method == args.reference:
            continue
        if len(values[method]) < 2:
            raise ValueError(f"Welch's test requires at least two runs for {method}.")
        statistic, p_value = ttest_ind(
            values[args.reference], values[method], equal_var=False,
            alternative="two-sided")
        comparisons.append({
            "reference": args.reference,
            "comparison": method,
            "metric": args.metric,
            "reference_seeds": seeds[args.reference],
            "comparison_seeds": seeds[method],
            "reference_values": values[args.reference],
            "comparison_values": values[method],
            "welch_t": float(statistic),
            "two_sided_p": float(p_value),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"comparisons": comparisons}, handle, indent=2)
    print(f"Saved {len(comparisons)} comparison(s) to {output}")


if __name__ == "__main__":
    main()

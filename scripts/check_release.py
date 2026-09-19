"""CPU-only static checks for a LAMP source release."""

import ast
import importlib
import json
import re
import sys
from pathlib import Path

import yaml


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIGS = {
    "configs/lamp.yaml", "configs/baseline.yaml", "configs/isic2018.yaml",
    "configs/busi.yaml", "configs/ablation/spb_only.yaml",
    "configs/ablation/w_o_spb.yaml", "configs/ablation/w_o_erab.yaml",
    "configs/ablation/w_o_afrb.yaml", "configs/ablation/entropy.yaml",
    "configs/ablation/confidence.yaml",
    "configs/ablation/reverse_attention.yaml",
    "configs/ablation/spb_groups_2.yaml",
    "configs/ablation/spb_groups_3.yaml",
    "configs/ablation/spb_groups_5.yaml",
    "configs/ablation/spb_parallel.yaml",
    "configs/ablation/spb_finest.yaml",
    "configs/ablation/fusion_add.yaml",
    "configs/ablation/fusion_concat.yaml",
    "configs/ablation/fusion_attention_gate.yaml",
    "configs/ablation/fusion_afrb.yaml",
    "configs/ablation/spb_stage6.yaml",
    "configs/ablation/spb_stages_5_6.yaml",
    "configs/ablation/spb_encoder_only.yaml",
    "configs/ablation/spb_decoder_only.yaml",
    "configs/ablation/erab_stage1.yaml",
    "configs/ablation/erab_stages_1_2.yaml",
    "configs/ablation/erab_stages_1_4.yaml",
    "configs/ablation/erab_wo_3x3.yaml",
    "configs/ablation/afrb_stage1.yaml",
    "configs/ablation/afrb_stages_1_2.yaml",
    "configs/ablation/afrb_stages_1_4.yaml",
    "configs/ablation/refinement_none.yaml",
    "configs/ablation/refinement_stages_2_3.yaml",
    "configs/ablation/refinement_stages_1_3.yaml",
    "configs/ablation/channels_small.yaml",
    "configs/ablation/channels_large.yaml",
    "configs/ablation/global_mamba.yaml",
    "configs/ablation/global_pvm.yaml",
}
EXPECTED_WEIGHTS = {"ISIC2017.pth", "ISIC2018.pth", "BUSI.pth"}
ENTRY_MODULES = [
    "train", "test", "benchmark", "visualization.visualize_segmentation",
    "visualization.visualize_erab", "visualization.visualize_ablation",
    "visualization.visualize_learned_responses",
    "analysis.analyze_internal_responses", "analysis.plot_loss_curves",
    "analysis.rank_failure_cases", "analysis.welch_ttest",
]


class Checks:
    def __init__(self):
        self.errors = []

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)


def config_checks(checks):
    for relative in sorted(EXPECTED_CONFIGS):
        path = ROOT / relative
        checks.require(path.is_file(), f"Missing config: {relative}")
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            with path.open(encoding="utf-8") as handle:
                value = yaml.safe_load(handle)
            checks.require(isinstance(value, dict), f"YAML root is not a mapping: {path}")
        except Exception as error:
            checks.errors.append(f"Invalid YAML {path}: {error}")


def python_checks(checks):
    for path in sorted(ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
        except Exception as error:
            checks.errors.append(f"Python syntax error {path}: {error}")
    sys.path.insert(0, str(ROOT))
    for module in ENTRY_MODULES:
        try:
            importlib.import_module(module)
        except Exception as error:
            checks.errors.append(f"Cannot import {module}: {error}")


def markdown_checks(checks):
    for document in sorted(ROOT.glob("*.md")):
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            checks.require((document.parent / target).exists(),
                           f"Broken local link in {document.name}: {target}")


def qualitative_case_checks(checks):
    manifest = ROOT / "visualization" / "case_ids.json"
    checks.require(manifest.is_file(), "Missing qualitative case manifest")
    if not manifest.is_file():
        return
    try:
        cases = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as error:
        checks.errors.append(f"Invalid qualitative case manifest: {error}")
        return
    split_files = {
        "ISIC2017": ROOT / "splits" / "isic2017_test.txt",
        "ISIC2018": ROOT / "splits" / "isic2018_test.txt",
        "BUSI": ROOT / "splits" / "BUSI_test.txt",
    }
    for figure, entries in cases.items():
        checks.require(isinstance(entries, list), f"Case list is not an array: {figure}")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            dataset = entry.get("dataset")
            checks.require(dataset in split_files, f"Unknown case dataset: {dataset}")
            if dataset not in split_files:
                continue
            names = split_files[dataset].read_text(encoding="utf-8").splitlines()
            index = entry.get("index")
            checks.require(isinstance(index, int) and 0 <= index < len(names),
                           f"Invalid test index in {figure}: {index}")
            if isinstance(index, int) and 0 <= index < len(names):
                checks.require(Path(names[index]).stem == entry.get("image_id"),
                               f"Image ID/index mismatch in {figure}: {entry}")


def release_hygiene_checks(checks):
    absolute_marker = "/" + "home/"
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_dir():
            checks.require(path.name not in {"__pycache__", ".pytest_cache"},
                           f"Cache directory present: {relative}")
            continue
        if path.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".sh"}:
            try:
                checks.require(absolute_marker not in path.read_text(encoding="utf-8"),
                               f"Absolute personal path present: {relative}")
            except UnicodeDecodeError:
                pass
        if path.suffix == ".pth":
            checks.require(path.parent == ROOT / "pretrained_pth" and
                           path.name in EXPECTED_WEIGHTS,
                           f"Unexpected checkpoint: {relative}")
        checks.require(not (path.name.endswith("~") or path.suffix in {".tmp", ".swp"}),
                       f"Temporary file present: {relative}")
    present = {path.name for path in (ROOT / "pretrained_pth").glob("*.pth")}
    checks.require(present == EXPECTED_WEIGHTS,
                   f"Checkpoint set differs: expected {sorted(EXPECTED_WEIGHTS)}, got {sorted(present)}")


def main():
    checks = Checks()
    config_checks(checks)
    python_checks(checks)
    markdown_checks(checks)
    qualitative_case_checks(checks)
    release_hygiene_checks(checks)
    if checks.errors:
        print("Release check failed:")
        for error in checks.errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Release check passed.")


if __name__ == "__main__":
    main()

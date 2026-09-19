#!/usr/bin/env bash
set -euo pipefail

mode="${1:-list}"
seeds=(42 43 44)

# ISIC 2017 table variants only. Dataset-specific comparisons for ISIC 2018
# and BUSI are intentionally not implied by this script.
configs=(
  configs/baseline.yaml
  configs/ablation/spb_only.yaml
  configs/ablation/w_o_afrb.yaml
  configs/ablation/entropy.yaml
  configs/ablation/confidence.yaml
  configs/ablation/reverse_attention.yaml
  configs/ablation/spb_groups_2.yaml
  configs/ablation/spb_groups_3.yaml
  configs/ablation/spb_groups_5.yaml
  configs/ablation/spb_parallel.yaml
  configs/ablation/spb_finest.yaml
  configs/ablation/fusion_add.yaml
  configs/ablation/fusion_concat.yaml
  configs/ablation/fusion_attention_gate.yaml
  configs/lamp.yaml
)

print_groups() {
  echo "[progressive-component-ablation-tables-5-to-7]"
  printf '%s\n' configs/baseline.yaml configs/ablation/spb_only.yaml \
    configs/ablation/w_o_afrb.yaml configs/lamp.yaml
  echo "[unreliability-map]"
  printf '%s\n' configs/ablation/entropy.yaml configs/ablation/confidence.yaml \
    configs/ablation/reverse_attention.yaml configs/lamp.yaml
  echo "[spb-mechanism]"
  printf '%s\n' configs/ablation/spb_groups_2.yaml configs/ablation/spb_groups_3.yaml \
    configs/lamp.yaml configs/ablation/spb_groups_5.yaml \
    configs/ablation/spb_parallel.yaml configs/ablation/spb_finest.yaml
  echo "[fusion]"
  printf '%s\n' configs/ablation/fusion_add.yaml configs/ablation/fusion_concat.yaml \
    configs/ablation/fusion_attention_gate.yaml configs/lamp.yaml
}

if [[ "$mode" == "list" ]]; then
  print_groups
elif [[ "$mode" == "train" ]]; then
  for config in "${configs[@]}"; do
    name="$(basename "$config" .yaml)"
    for seed in "${seeds[@]}"; do
      python train.py --config "$config" --seed "$seed" \
        --output "results/tables_isic2017/${name}/seed_${seed}"
    done
  done
elif [[ "$mode" == "eval" ]]; then
  for config in "${configs[@]}"; do
    name="$(basename "$config" .yaml)"
    for seed in "${seeds[@]}"; do
      checkpoint="results/tables_isic2017/${name}/seed_${seed}/best.pth"
      output="results/tables_isic2017/${name}/seed_${seed}/test"
      if [[ -f "$checkpoint" ]]; then
        python test.py --config "$config" --checkpoint "$checkpoint" --output "$output"
      else
        echo "SKIP missing checkpoint: $checkpoint" >&2
      fi
    done
  done
else
  echo "Usage: bash scripts/reproduce_tables.sh [list|train|eval]" >&2
  exit 2
fi

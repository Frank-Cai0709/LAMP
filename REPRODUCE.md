# Reproducing the paper artifacts

Run all commands from the repository root after installing dependencies and
placing the datasets under `data/`. Commands below preserve the configured loss,
metric aggregation, augmentation, schedule, and random seeds.

## Directly supported experiments

| Experiment | Configuration | Command | Output | Mode |
|---|---|---|---|---|
| Progressive component ablation: Baseline | `configs/baseline.yaml` | `python train.py --config configs/baseline.yaml` | `results/baseline_isic2017/` | Training |
| Progressive component ablation: +SPB | `configs/ablation/spb_only.yaml` | `python train.py --config configs/ablation/spb_only.yaml` | `results/spb_only/` | Training |
| Progressive component ablation: +SPB+ERAB | `configs/ablation/w_o_afrb.yaml` | `python train.py --config configs/ablation/w_o_afrb.yaml` | `results/w_o_afrb/` | Training |
| Progressive component ablation: Full LAMP | `configs/lamp.yaml` | `python train.py --config configs/lamp.yaml` | `results/lamp_isic2017/` | Training or checkpoint evaluation |
| SPB group number | `spb_groups_2.yaml`, `spb_groups_3.yaml`, `lamp.yaml`, `spb_groups_5.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| SPB propagation | `spb_parallel.yaml`, `lamp.yaml`, `spb_finest.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| SPB depth placement (Table 10) | `spb_stage6.yaml`, `spb_stages_5_6.yaml`, `lamp.yaml` | `python train.py --config configs/ablation/<CONFIG>` | Configured result directory | Training |
| SPB encoder/decoder placement (Table 11) | `w_o_spb.yaml`, `spb_encoder_only.yaml`, `spb_decoder_only.yaml`, `lamp.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| ERAB placement and branch (Tables 13–14) | `erab_stage1.yaml`, `erab_stages_1_2.yaml`, `lamp.yaml`, `erab_stages_1_4.yaml`, `erab_wo_3x3.yaml` | `python train.py --config configs/ablation/<CONFIG>` | Configured result directory | Training |
| AFRB placement/refinement (Tables 15–16) | `afrb_stage1.yaml`, `afrb_stages_1_2.yaml`, `lamp.yaml`, `afrb_stages_1_4.yaml`, `refinement_*.yaml` | `python train.py --config configs/ablation/<CONFIG>` | Configured result directory | Training |
| Unreliability-map ablation | `entropy.yaml`, `confidence.yaml`, `reverse_attention.yaml`, `lamp.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| Fusion strategy | `fusion_add.yaml`, `fusion_concat.yaml`, `fusion_attention_gate.yaml`, `lamp.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| Channel widths (Table 19) | `channels_small.yaml`, `lamp.yaml`, `channels_large.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| Mamba/PVM/SPB (Tables 20–22) | `global_mamba.yaml`, `global_pvm.yaml`, `lamp.yaml` | `python train.py --config <CONFIG>` | Configured result directory | Training |
| End-to-end timing | `configs/lamp.yaml` | `python benchmark.py --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth --output results/benchmark` | `results/benchmark/benchmark.json` and `composites/` | Inference |
| Segmentation visualization | `configs/lamp.yaml` | `python visualization/visualize_segmentation.py --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth --image <IMAGE> --output results/segmentation.png` | Selected output image | Inference |
| Unreliability visualization | `configs/lamp.yaml` | `python visualization/visualize_erab.py --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth --image <IMAGE> --output results/erab_maps.png` | Selected output image | Inference |

Evaluate a released checkpoint:

```bash
python test.py --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth \
  --output results/released_isic2017
```

For the ISIC 2017 three-seed ablations:

```bash
bash scripts/reproduce_tables.sh list
bash scripts/reproduce_tables.sh train
bash scripts/reproduce_tables.sh eval
```

The script runs the ISIC 2017 experiments with seeds 42, 43, and 44 and writes
under `results/tables_isic2017/<variant>/seed_<seed>/`.

Tables 5–7 follow the progressive sequence Baseline → +SPB → +SPB+ERAB → Full
LAMP.

For Table 10, a numbered semantic stage is treated symmetrically: Stage 6 is
`encoder6 + decoder1`, Stage 5–6 adds `encoder5 + decoder2`, and Stage 4–6 is
the default `encoder4–6 + decoder1–3`. This mapping makes the Table 10 full
setting identical to the released LAMP architecture.

## Other datasets

```bash
python train.py --config configs/isic2018.yaml
python test.py --config configs/isic2018.yaml \
  --checkpoint pretrained_pth/ISIC2018.pth

python train.py --config configs/busi.yaml
```

## Lightweight post-hoc analysis

Welch's two-sided t-test can be computed from saved per-seed scalar results:

```bash
python analysis/welch_ttest.py --input results/dsc_runs.csv \
  --reference LAMP --metric DSC --output results/welch_ttest.json
```

The input CSV contains `method`, `seed`, and the selected metric column.

Feature discrepancy, top-5% boundary-error concentration, and correction
intensity are computed from a checkpoint and the configured test split:

```bash
python analysis/analyze_internal_responses.py \
  --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth \
  --output results/internal_responses
```

The command writes `internal_responses.csv` (one row per test image) and
`internal_responses.json` (mean and population standard deviation). The top-5%
statistic uses pixelwise segmentation errors inside a 3x3 ground-truth boundary
band as its denominator. Correction intensity uses the absolute value of
`U_i * g_i^s * tanh(s_i-d_i)`; low and high regions are split at each map's
median.

Loss figures can be regenerated from an actual `training.jsonl`:

```bash
python analysis/plot_loss_curves.py --log results/lamp_isic2017/training.jsonl \
  --output results/loss_curve_isic2017.pdf --title "ISIC 2017"
```

The exact Figure 11 and Figure 13 samples, including original image IDs and NPY
test indices, are recorded in `visualization/case_ids.json`. The Figure 11
layout can be generated from a recorded case position;
the Grad-CAM target is the mean predicted foreground probability and the hooked
feature is `encoder6`, the third encoder SPB:

```bash
python visualization/visualize_learned_responses.py \
  --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth \
  --case-position 0 --output results/learned_response.png
```

Failure cases can be selected reproducibly by ascending per-image DSC:

```bash
python analysis/rank_failure_cases.py --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth \
  --output results/failure_case_ranking.csv
```

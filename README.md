# LAMP: Towards Efficient Medical Image Segmentation via Decoupled Perception and Adaptive Reconciliation

Official implementation of **Lightweight Adaptive Multi-perspective Perception Network (LAMP)** for medical image segmentation. The repository provides configuration-driven training, evaluation, ablation studies, and qualitative visualization while preserving the model used for the reported experiments.

## Method

LAMP combines three components:

- **SPB** performs hierarchical multi-granularity semantic propagation with grouped state-space modeling.
- **ERAB** derives a sinusoidal unreliability response,
  `u = sin(π × sigmoid(g))`, and uses it to refine boundary-sensitive skip
  features.
- **AFRB** adaptively fuses decoder and skip features, including discrepancy-aware recalibration at the configured fusion stage.

The default model uses four SPB groups, sequential propagation while preserving
the finest group, ERAB maps, and AFRB fusion at three decoder stages. Group-count
ablations keep the backbone channels fixed; when an SPB input width is not
divisible by the selected group count, a token-wise linear projection expands
it to the smallest compatible internal width.

## Repository layout

```text
LAMP/
├── configs/                 # Dataset, model, and ablation YAML files
├── datasets/                # NPY dataset loaders
├── models/                  # LAMP, baseline, and reusable modules
├── scripts/                 # Training, testing, and table reproduction
├── splits/                  # Readable split metadata when recoverable
├── utils/                   # Losses, metrics, configuration, utilities
├── visualization/           # Qualitative figure commands
├── train.py
├── test.py
├── benchmark.py
├── requirements.txt
└── environment.yml
```

See [REPRODUCE.md](REPRODUCE.md) for the paper experiment-to-command mapping.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate lamp
```

Alternatively, install the Python dependencies in an existing CUDA-enabled PyTorch environment:

```bash
pip install -r requirements.txt
```

`environment.yml` and `requirements.txt` describe the **recommended
compatibility environment** for this release. 

## Compute environment

### Paper experiment environment

The following values describe the workstation and `vmunet` Conda environment
used for all reported experiments. Although the workstation contains two NVIDIA
GeForce RTX 4090 GPUs, every experiment was performed using a single GPU.

| Item | Reported setup |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| CPU | Intel Core i9-14900KF |
| RAM | 192 GB installed (approximately 188 GiB usable) |
| GPU | 2 × NVIDIA GeForce RTX 4090 (24 GB each) |
| Python | 3.8.20 |
| PyTorch | 1.13.0+cu117 |
| CUDA | 11.7 |
| cuDNN | 8.5.0 |
| Input resolution | 256 × 256 |
| Training batch size | 8 |
| Inference batch size | 1 |
| DataLoader workers | 0 |

### Recommended compatibility environment

The supplied `environment.yml` currently specifies Python 3.10, PyTorch 2.0 or
newer, torchvision 0.15 or newer, and CUDA 11.8. Exact dependency constraints
are listed in `requirements.txt`. This environment is intended for practical
reproduction and is not asserted to be identical to the table-producing setup.

## Dataset preparation

Set `dataset.root` in the selected YAML file. Each dataset directory uses the following arrays:

```text
data_train.npy   mask_train.npy
data_val.npy     mask_val.npy
data_test.npy    mask_test.npy
```

Images are expected in `N × H × W × 3` form and masks in `N × H × W` form.
Dataset-specific split counts and provenance are provided in
[datasets/README.md](datasets/README.md). Exact ISIC and BUSI source filename
assignments are provided in [splits/README.md](splits/README.md).

Weights download: [Google Drive](https://drive.google.com/drive/folders/1M5bHgyesni65gWvYhJG5RKJ6KEMMn0Bk?usp=drive_link)

Complete dataset package: [Baidu Netdisk](https://pan.baidu.com/s/1udBcqarU5HlEmz-23VkXbw?pwd=2026)

- File: `LAMP_datasets_full.zip`
- Extraction code: `2026`
- Contents: BUSI, ISIC 2017, and ISIC 2018, including the prepared NPY splits used by this repository.

After downloading, keep the dataset directories under `data/` as follows:

```text
data/
├── dataset_isic17/
│   ├── data_train.npy
│   ├── data_val.npy
│   ├── data_test.npy
│   ├── mask_train.npy
│   ├── mask_val.npy
│   └── mask_test.npy
├── dataset_isic18/
│   └── ... same six split files
└── BUSI/
    └── seg/
        └── ... same six split files
```

Place downloaded model files in `pretrained_pth/`. No source-code change is required after setting the dataset root.

## Training

```bash
python train.py --config configs/lamp.yaml
```

Training uses the paper loss protocol, `BCE + Dice loss`, with unit weights,
Dice smoothing equal to 1, and `BCELoss` applied directly to the predicted
probabilities. It uses a fixed seed and writes `latest.pth`, `best.pth`, periodic
snapshots, and `training.jsonl` to the configured output directory. Following
the paper protocol, `best.pth` is selected by the minimum validation BCE-Dice
loss. Validation DSC and the other segmentation metrics are recorded alongside
the selection loss. Resume a run with:

```bash
python train.py --config configs/lamp.yaml --resume results/lamp_isic2017/latest.pth
```

For the three-run protocol reported in the ablation tables, use seeds 42, 43, and 44, for example:

```bash
python train.py --config configs/lamp.yaml --seed 43 --output results/lamp/seed_43
```

ISIC 2018 and BUSI use the same entry point:

```bash
python train.py --config configs/isic2018.yaml
python train.py --config configs/busi.yaml
```

## Evaluation

```bash
python test.py --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth
```

Evaluation reports DSC, IoU, sensitivity (SE), specificity (SP), accuracy (ACC), and precision. It saves `metrics.json` and probability maps under `predictions/`.

## Inference speed

The timed region covers the complete evaluation loop used in the reference implementation, including batch retrieval, host-to-device transfer, model inference, prediction post-processing, result generation, and final synchronization.

The default batch size is 1, `num_workers` is 0, and PNG compression level is 8.
This end-to-end definition matches the timing protocol used for the reported
result.

Run the same end-to-end protocol used for the approximately `0.027 s/image` measurement with:

```bash
python benchmark.py --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth \
  --output results/benchmark
```

The command saves `benchmark.json` and generated test composites. The JSON
records GPU, CPU, RAM, OS/platform, Python, PyTorch, CUDA, cuDNN, image size,
batch size, worker count, timing mode, evaluated image count, and checkpoint
path. End-to-end throughput is computed as images divided by total timed seconds
(`1 / 0.027`, approximately 37 images/s for the reported value).

## Reproduce experiments

All variants inherit from `configs/lamp.yaml`; an ablation file changes only the relevant option.

| Paper experiment | Configuration(s) |
|---|---|
| Full LAMP and baseline | `lamp.yaml`, `baseline.yaml` |
| Progressive component ablation (Tables 5–7) | `baseline.yaml` → `spb_only.yaml` → `w_o_afrb.yaml` → `lamp.yaml` |
| Unreliability maps | `entropy.yaml`, `confidence.yaml`, `reverse_attention.yaml`, `lamp.yaml` |
| SPB group number | `spb_groups_2.yaml`, `spb_groups_3.yaml`, `lamp.yaml`, `spb_groups_5.yaml` |
| SPB propagation | `spb_parallel.yaml`, `lamp.yaml`, `spb_finest.yaml` |
| Fusion comparison | `fusion_add.yaml`, `fusion_concat.yaml`, `fusion_attention_gate.yaml`, `fusion_afrb.yaml` |

List or train all table variants:

```bash
bash scripts/reproduce_tables.sh list
bash scripts/reproduce_tables.sh train
bash scripts/reproduce_tables.sh eval
```

This helper runs the ISIC 2017 variants with seeds 42, 43, and 44. See
`REPRODUCE.md` for the complete experiment-to-command mapping.

The component tables use progressive addition: Encoder–Decoder baseline,
`+SPB`, `+SPB+ERAB`, and finally `+SPB+ERAB+AFRB`. The auxiliary
`w_o_spb.yaml` and `w_o_erab.yaml` files are not presented as reproductions of
Tables 5–7.

## Visualization

Segmentation figure:

```bash
python visualization/visualize_segmentation.py --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth --image examples/sample.jpg
```

Stage-specific ERAB response figure:

```bash
python visualization/visualize_erab.py --config configs/lamp.yaml \
  --checkpoint pretrained_pth/ISIC2017.pth --image examples/sample.jpg \
  --stage 3 --output results/erab_stage3.png
```

Ablation panel from previously generated predictions:

```bash
python visualization/visualize_ablation.py \
  --images results/add.png results/concat.png results/afrb.png \
  --labels Add Concat AFRB --output results/ablation.png
```

## Configuration

YAML files control the dataset, image size, batch size, epochs, optimizer, learning rate, scheduler, augmentation, and every supported model variant. Key switches are `model.spb`, `model.erab`, and `model.afrb`. New experiments should inherit the default file through `_base_` and override only the intended field.

## Release sanity check

Run the CPU-only static release audit before publishing:

```bash
python scripts/check_release.py
```


## License
Citation information will be updated after publication.

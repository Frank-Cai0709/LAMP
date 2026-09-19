#!/usr/bin/env bash
set -euo pipefail
python test.py --config configs/lamp.yaml --checkpoint pretrained_pth/ISIC2017.pth "$@"

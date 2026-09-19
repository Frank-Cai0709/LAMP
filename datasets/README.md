# Dataset layout and splits

Place the NumPy split files under the directory configured by `dataset.root`:

```text
data_train.npy  mask_train.npy
data_val.npy    mask_val.npy
data_test.npy   mask_test.npy
```

Images use `N x H x W x 3`; masks use `N x H x W`. The loaders convert masks
with `mask > 0` and reproduce the dataset-level normalization and augmentations
used in the paper.

The default configuration paths are `data/dataset_isic17`,
`data/dataset_isic18`, and `data/BUSI/seg`. The NPY files, rather than the
loader, define the sample assignment.

## Actual split sizes

| Dataset | Training | Validation | Testing | Total used |
|---|---:|---:|---:|---:|
| ISIC 2017 | 1,250 | 150 | 600 | 2,000 |
| ISIC 2018 | 1,815 | 259 | 520 | 2,594 |
| BUSI | 453 | 65 | 129 | 647 |

ISIC 2017 uses the 2,000 prepared images assigned as 1,250/150/600. ISIC 2018
uses all 2,594 prepared images assigned as 1,815/259/520. Pixel-level hashes of
the currently distributed arrays show no identical image across training,
validation, and testing subsets. Exact source filename assignments and their
zero-based NPY indices are provided under `splits/`; see `splits/README.md` for
the content-matching verification method.

## BUSI protocol

The original BUSI collection contains 780 images: 437 benign, 210 malignant,
and 133 normal images. The segmentation experiments use the 647 lesion images
from the benign and malignant categories. The released assignment preserves
the validation and testing subsets used by the retained BUSI experiment. The
training subset is the complement of those two subsets, so all three subsets
are mutually exclusive. Exact filename assignments are supplied under
`splits/` and should be used instead of relying on filesystem enumeration.

| Subset | Benign | Malignant | Total |
|---|---:|---:|---:|
| Training | 304 | 149 | 453 |
| Validation | 44 | 21 | 65 |
| Testing | 89 | 40 | 129 |

The readable BUSI filename assignments are provided under `splits/`.

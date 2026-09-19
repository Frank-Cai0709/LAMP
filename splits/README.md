# Split manifests

The BUSI text files contain the exact filename assignment for the currently
distributed split. The validation and testing assignments preserve the
retained BUSI experiment, while training contains the remaining images. Each
line is a source image filename; masks follow the corresponding BUSI naming
convention. Use these manifests instead of filesystem enumeration when
preparing the arrays on another machine.

The ISIC text files contain the source image filenames in the exact order of
each distributed NPY split. The accompanying `isic2017_manifest.csv` and
`isic2018_manifest.csv` files additionally record the zero-based NPY index,
source mask filename, exact resized-mask SHA-256, and image verification
statistics for every sample.

The ISIC assignments were recovered against the retained official source files
by requiring a unique, pixel-exact match between every NPY mask and the
corresponding resized source mask. The associated source image was then checked
against the NPY image; across both datasets the minimum pixel correlation is
`0.997928064` and the maximum mean absolute error is `0.983907064` on the
0--255 scale. Run `scripts/recover_isic_split_manifests.py` to reproduce these
files when the original ISIC image and mask directories are available.

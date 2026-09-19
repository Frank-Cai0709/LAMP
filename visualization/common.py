from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_image(path, size, device):
    image = Image.open(path).convert("RGB").resize(tuple(size[::-1]), Image.BILINEAR)
    array = np.asarray(image).astype(np.float32)
    array = (array - array.mean()) / max(float(array.std()), 1e-8)
    array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8) * 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return image, tensor


def save_panel(images, titles, output, cmap=None):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    axes = np.atleast_1d(axes)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap=cmap if np.asarray(image).ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

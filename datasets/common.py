from pathlib import Path
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def normalize_dataset(images):
    images = images.astype(np.float32)
    normalized = (images - images.mean()) / max(float(images.std()), 1e-8)
    output = np.empty_like(normalized)
    for index, image in enumerate(normalized):
        low, high = float(image.min()), float(image.max())
        output[index] = (image - low) / max(high - low, 1e-8) * 255.0
    return output


class NpySegmentationDataset(Dataset):
    def __init__(self, root, split, augment=False, image_size=None):
        root = Path(root)
        self.images = normalize_dataset(np.load(root / f"data_{split}.npy"))
        self.masks = (np.load(root / f"mask_{split}.npy") > 0).astype(np.float32)
        self.augment = augment
        if image_size and tuple(self.images.shape[1:3]) != tuple(image_size):
            raise ValueError(
                f"Split size {self.images.shape[1:3]} does not match configured image_size {image_size}. "
                "Prepare the arrays at the configured resolution."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        enabled = bool(self.augment.get("enabled", True)) if isinstance(self.augment, dict) else bool(self.augment)
        flip_p = self.augment.get("rot90_flip_probability", 0.5) if isinstance(self.augment, dict) else 0.5
        rotation_p = self.augment.get("free_rotation_probability", 0.5) if isinstance(self.augment, dict) else 0.5
        degrees = self.augment.get("free_rotation_degrees", [20, 79]) if isinstance(self.augment, dict) else [20, 79]
        if enabled and random.random() < flip_p:
            k = np.random.randint(0, 4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis).copy()
            mask = np.flip(mask, axis).copy()
        if enabled and random.random() < rotation_p:
            angle = np.random.randint(int(degrees[0]), int(degrees[1]) + 1)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.asarray(mask).copy()).unsqueeze(0).float()
        return image, mask, index

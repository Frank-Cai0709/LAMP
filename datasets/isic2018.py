from .common import NpySegmentationDataset


def build_dataset(root, split, augment=False, image_size=None):
    return NpySegmentationDataset(root, split, augment, image_size)

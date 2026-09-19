from importlib import import_module


def build_dataset(config, split, augment=False):
    module = import_module(f"datasets.{config['name'].lower()}")
    return module.build_dataset(config["root"], split, augment, config.get("image_size"))

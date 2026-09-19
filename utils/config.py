from copy import deepcopy
from pathlib import Path
import yaml


def _merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base = config.pop("_base_", None)
    if base:
        base_config = load_config(path.parent / base)
        config = _merge(base_config, config)
    config["config_path"] = str(path)
    return config

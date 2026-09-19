import json
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_checkpoint(model, path, device="cpu"):
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if any(key.startswith("module.") for key in state):
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return payload

"""Shared reproducibility, device and checkpoint helpers."""
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    return device


def save_checkpoint(path: Path, model, model_name: str, class_names: Dict[int, str],
                    image_size: int, history=None, metrics=None) -> None:
    """Save weights plus inference metadata; no results are fabricated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
        "history": history or {},
        "metrics": metrics or {},
    }, path)

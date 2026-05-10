from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from . import config


def ensure_dirs() -> None:
    for path in [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.SPLIT_DIR,
        config.CHECKPOINT_DIR,
        config.METRICS_DIR,
        config.FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str | None = None):
    if torch is None:
        raise ImportError("torch is required for model training/evaluation. Run `pip install -r requirements.txt`.")
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}\nExpected path: {path}")


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

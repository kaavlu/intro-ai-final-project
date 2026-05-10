from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

FOODNEXTDB_DIR = RAW_DATA_DIR / "FoodNExTDB"
FOOD41_DIR = RAW_DATA_DIR / "food41"
NUTRITION_DIR = RAW_DATA_DIR / "nutrition"

METADATA_CSV = PROCESSED_DATA_DIR / "foodnextdb_metadata.csv"
FILTERED_METADATA_CSV = PROCESSED_DATA_DIR / "foodnextdb_filtered.csv"
MAPPED_METADATA_CSV = PROCESSED_DATA_DIR / "foodnextdb_with_calories.csv"
SPLIT_DIR = PROCESSED_DATA_DIR / "splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
MIN_SUBCATEGORY_SAMPLES = 100
MAX_SAMPLES = 6000

TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

PORTION_MULTIPLIER = 1.0
FUZZY_MATCH_THRESHOLD = 85

LABEL_COLUMNS = ["category", "subcategory", "cooking_style"]
REQUIRED_METADATA_COLUMNS = ["image_path", *LABEL_COLUMNS]

LOSS_WEIGHT_CONFIGS = [
    (1.0, 1.0, 1.0, 0.5),
    (1.0, 2.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 2.0),
]

AUGMENTATION_CONFIGS = {
    "resize": "baseline resize",
    "flip_color": "horizontal flip + color jitter",
    "crop_rotate": "random resized crop + rotation",
}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "multitask_resnet18"
    augmentation: str = "resize"
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    loss_weights: tuple[float, float, float, float] = LOSS_WEIGHT_CONFIGS[0]
    regression_loss: str = "smooth_l1"
    device: str | None = None
    # When set, only the first N rows of train/dev are used (CPU smoke runs).
    smoke_max_samples: int | None = None

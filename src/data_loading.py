from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .utils import ensure_dirs, require_path, save_json, set_seed

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    train_test_split = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    transforms = None

    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOOD41_IGNORE_DIRS = {"meta", "__macosx"}


def _clean_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null"}:
        return None
    return text


def _mode_or_first(series: pd.Series) -> str | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    modes = cleaned.mode()
    return str(modes.iloc[0] if not modes.empty else cleaned.iloc[0])


def _humanize_class_name(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def _candidate_image_roots(raw_dir: Path) -> list[Path]:
    candidates = [
        raw_dir / "images",
        raw_dir / "food-101" / "images",
        raw_dir / "Food-101" / "images",
        raw_dir,
    ]
    return [path for path in candidates if path.exists()]


def _looks_like_class_dir(path: Path) -> bool:
    if path.name.lower() in FOOD41_IGNORE_DIRS:
        return False
    return any(path.glob("*.jpg")) or any(path.glob("*.jpeg")) or any(path.glob("*.png"))


def scan_food41(raw_dir: Path = config.FOOD41_DIR) -> pd.DataFrame:
    require_path(
        raw_dir,
        "Food41 data was not found. Download and extract the Kaggle archive before scanning.",
    )
    rows: list[dict[str, str]] = []
    image_root = None
    for candidate in _candidate_image_roots(raw_dir):
        class_dirs = [
            path
            for path in sorted(candidate.iterdir())
            if path.is_dir() and _looks_like_class_dir(path)
        ]
        if class_dirs:
            image_root = candidate
            break

    if image_root is None:
        raise FileNotFoundError(
            f"No Food41 class image folders were found under {raw_dir}. "
            "Expected folders such as data/raw/food41/images/apple_pie/*.jpg."
        )

    for class_dir in class_dirs:
        subcategory = _humanize_class_name(class_dir.name)
        image_paths = []
        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(class_dir.glob(f"*{extension}"))
        for image_path in sorted(image_paths):
            rows.append(
                {
                    "image_path": str(image_path),
                    "category": "food",
                    "subcategory": subcategory,
                    "cooking_style": "unknown",
                }
            )

    df = pd.DataFrame(rows, columns=config.REQUIRED_METADATA_COLUMNS)
    print_dataset_stats(df, title="Food41 metadata")
    print("Food41 provides dish class folders only; category='food' and cooking_style='unknown' are derived placeholders.")
    return df


def scan_foodnextdb(raw_dir: Path) -> pd.DataFrame:
    require_path(raw_dir, "FoodNExTDB was not found. Download/extract it before scanning.")
    rows: list[dict[str, str]] = []
    missing_images = 0
    missing_labels = 0
    participant_dirs = [p for p in raw_dir.rglob("A4F_*") if p.is_dir()]

    if not participant_dirs:
        raise FileNotFoundError(f"No participant folders matching A4F_* were found under {raw_dir}.")

    for participant_dir in sorted(participant_dirs):
        csv_files = sorted(participant_dir.glob("*_labeled_data.csv"))
        if not csv_files:
            continue
        image_paths = [p for p in participant_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        image_lookup = {p.name.lower(): p for p in image_paths}
        image_lookup.update({p.stem.lower(): p for p in image_paths})

        for csv_path in csv_files:
            labeled = pd.read_csv(csv_path)
            missing_columns = {"id", *config.LABEL_COLUMNS} - set(labeled.columns)
            if missing_columns:
                warnings.warn(f"Skipping {csv_path}: missing columns {sorted(missing_columns)}")
                continue

            labeled = labeled[["id", *config.LABEL_COLUMNS]].copy()
            for col in config.LABEL_COLUMNS:
                labeled[col] = labeled[col].map(_clean_label)
            missing_labels += int(labeled[config.LABEL_COLUMNS].isna().any(axis=1).sum())
            labeled = labeled.dropna(subset=config.LABEL_COLUMNS)
            labeled = (
                labeled.groupby("id", as_index=False)
                .agg({col: _mode_or_first for col in config.LABEL_COLUMNS})
                .dropna(subset=config.LABEL_COLUMNS)
            )

            for record in labeled.to_dict("records"):
                image_id = str(record["id"]).strip()
                image_path = image_lookup.get(image_id.lower()) or image_lookup.get(
                    Path(image_id).stem.lower()
                )
                if image_path is None or not image_path.exists():
                    missing_images += 1
                    continue

                rows.append(
                    {
                        "image_path": str(image_path),
                        "category": record["category"],
                        "subcategory": record["subcategory"],
                        "cooking_style": record["cooking_style"],
                    }
                )

    df = pd.DataFrame(rows, columns=config.REQUIRED_METADATA_COLUMNS)
    print_dataset_stats(df, title="FoodNExTDB metadata")
    print(f"Dropped rows with missing images: {missing_images}")
    print(f"Dropped rows with missing labels: {missing_labels}")
    return df


def print_dataset_stats(df: pd.DataFrame, title: str = "Dataset") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Total samples: {len(df)}")
    for col in config.LABEL_COLUMNS:
        if col in df.columns:
            print(f"{col} classes: {df[col].nunique()}")
            print(df[col].value_counts().head(20).to_string())
            print()


def save_metadata(df: pd.DataFrame, path: Path = config.METADATA_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved metadata to {path}")


def load_metadata(path: Path = config.METADATA_CSV) -> pd.DataFrame:
    require_path(path, "Processed metadata was not found. Run the data scan first.")
    return pd.read_csv(path)


def filter_and_sample(
    df: pd.DataFrame,
    min_subcategory_samples: int = config.MIN_SUBCATEGORY_SAMPLES,
    max_samples: int = config.MAX_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    counts = df["subcategory"].value_counts()
    keep_subcategories = counts[counts >= min_subcategory_samples].index
    filtered = df[df["subcategory"].isin(keep_subcategories)].copy()

    if filtered.empty:
        raise ValueError(
            "Filtering removed every row. Lower MIN_SUBCATEGORY_SAMPLES or inspect labels."
        )

    if len(filtered) > max_samples:
        per_class_target = max(1, max_samples // filtered["subcategory"].nunique())
        sampled_parts = []
        for _, group in filtered.groupby("subcategory"):
            n = min(len(group), per_class_target)
            sampled_parts.append(group.sample(n=n, random_state=seed))
        sampled = pd.concat(sampled_parts)

        if len(sampled) < max_samples:
            remainder = filtered.drop(sampled.index, errors="ignore")
            if not remainder.empty:
                extra = remainder.sample(
                    n=min(max_samples - len(sampled), len(remainder)),
                    random_state=seed,
                )
                sampled = pd.concat([sampled, extra])
        filtered = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print_dataset_stats(filtered, title="Filtered food image metadata")
    return filtered


def create_splits(
    df: pd.DataFrame,
    split_dir: Path = config.SPLIT_DIR,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_test_split is None:
        raise ImportError("scikit-learn is required for stratified splits. Run `pip install -r requirements.txt`.")
    split_dir.mkdir(parents=True, exist_ok=True)

    stratify = df["subcategory"] if df["subcategory"].value_counts().min() >= 2 else None
    if stratify is None:
        warnings.warn("Some subcategories have fewer than 2 samples; splitting without stratify.")

    train_df, temp_df = train_test_split(
        df,
        train_size=config.TRAIN_RATIO,
        random_state=seed,
        stratify=stratify,
    )

    temp_stratify = (
        temp_df["subcategory"] if temp_df["subcategory"].value_counts().min() >= 2 else None
    )
    relative_dev_ratio = config.DEV_RATIO / (config.DEV_RATIO + config.TEST_RATIO)
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=relative_dev_ratio,
        random_state=seed,
        stratify=temp_stratify,
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    dev_df.to_csv(split_dir / "dev.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    print(f"Train/dev/test sizes: {len(train_df)} / {len(dev_df)} / {len(test_df)}")
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_label_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        col: {label: idx for idx, label in enumerate(sorted(df[col].dropna().unique()))}
        for col in config.LABEL_COLUMNS
    }


def save_label_maps(label_maps: dict[str, dict[str, int]], path: Path | None = None) -> None:
    save_json(label_maps, path or config.PROCESSED_DATA_DIR / "label_maps.json")


def get_transforms(augmentation: str = "resize", train: bool = True) -> Callable:
    if transforms is None:
        raise ImportError("torchvision is required for image transforms. Run `pip install -r requirements.txt`.")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if not train or augmentation == "resize":
        return transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if augmentation == "flip_color":
        return transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if augmentation == "crop_rotate":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.75, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                normalize,
            ]
        )
    raise ValueError(f"Unknown augmentation config: {augmentation}")


class FoodImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_maps: dict[str, dict[str, int]],
        transform: Callable | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.label_maps = label_maps
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if torch is None:
            raise ImportError("torch is required for FoodImageDataset. Run `pip install -r requirements.txt`.")
        if Image is None:
            raise ImportError("Pillow is required for image loading. Run `pip install -r requirements.txt`.")
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        targets = {
            "category": torch.tensor(
                self.label_maps["category"][row["category"]], dtype=torch.long
            ),
            "subcategory": torch.tensor(
                self.label_maps["subcategory"][row["subcategory"]], dtype=torch.long
            ),
            "cooking_style": torch.tensor(
                self.label_maps["cooking_style"][row["cooking_style"]], dtype=torch.long
            ),
            "calories": torch.tensor(float(row.get("calories", 0.0)), dtype=torch.float32),
        }
        return image, targets


def make_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_maps: dict[str, dict[str, int]],
    augmentation: str = "resize",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:
    if DataLoader is None:
        raise ImportError("torch is required for DataLoader creation. Run `pip install -r requirements.txt`.")
    return {
        "train": DataLoader(
            FoodImageDataset(train_df, label_maps, get_transforms(augmentation, train=True)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "dev": DataLoader(
            FoodImageDataset(dev_df, label_maps, get_transforms("resize", train=False)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            FoodImageDataset(test_df, label_maps, get_transforms("resize", train=False)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def prepare_foodnextdb_metadata() -> pd.DataFrame:
    ensure_dirs()
    df = scan_foodnextdb(config.FOODNEXTDB_DIR)
    save_metadata(df, config.METADATA_CSV)
    filtered = filter_and_sample(df)
    save_metadata(filtered, config.FILTERED_METADATA_CSV)
    create_splits(filtered)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare food image metadata and splits.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=config.METADATA_CSV)
    parser.add_argument("--dataset", choices=["food41", "foodnextdb"], default="foodnextdb")
    args = parser.parse_args()

    try:
        set_seed()
        ensure_dirs()
        raw_dir = args.raw_dir or (
            config.FOOD41_DIR if args.dataset == "food41" else config.FOODNEXTDB_DIR
        )
        df = scan_food41(raw_dir) if args.dataset == "food41" else scan_foodnextdb(raw_dir)
        save_metadata(df, args.output)
        filtered = filter_and_sample(df)
        save_metadata(filtered, config.FILTERED_METADATA_CSV)
        create_splits(filtered)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()

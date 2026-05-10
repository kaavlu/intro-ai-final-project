from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from . import config
from .data_loading import (
    build_label_maps,
    create_splits,
    make_dataloaders,
    save_label_maps,
)
from .models import build_model
from .nutrition_mapping import add_calorie_targets
from .utils import ensure_dirs, get_device, require_path, save_json, set_seed


def _load_training_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.MAPPED_METADATA_CSV.exists():
        if not config.FILTERED_METADATA_CSV.exists():
            raise FileNotFoundError(
                "Prepared metadata was not found. Run `python -m src.data_loading` first."
            )
        add_calorie_targets()

    mapped = pd.read_csv(config.MAPPED_METADATA_CSV)
    train_path = config.SPLIT_DIR / "train.csv"
    dev_path = config.SPLIT_DIR / "dev.csv"
    test_path = config.SPLIT_DIR / "test.csv"

    if all(path.exists() for path in [train_path, dev_path, test_path]):
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        if "calories" in train_df.columns:
            return train_df, dev_df, test_df

    return create_splits(mapped)


def _losses(outputs: torch.Tensor | dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"subcategory": nn.functional.cross_entropy(outputs, targets["subcategory"])}

    regression_target = targets["calories"].float()
    return {
        "category": nn.functional.cross_entropy(outputs["category"], targets["category"]),
        "subcategory": nn.functional.cross_entropy(outputs["subcategory"], targets["subcategory"]),
        "cooking_style": nn.functional.cross_entropy(
            outputs["cooking_style"], targets["cooking_style"]
        ),
        "calories": nn.functional.smooth_l1_loss(outputs["calories"], regression_target),
    }


def weighted_total_loss(
    losses: dict[str, torch.Tensor],
    weights: tuple[float, float, float, float],
) -> torch.Tensor:
    if set(losses) == {"subcategory"}:
        return losses["subcategory"]
    return (
        weights[0] * losses["category"]
        + weights[1] * losses["subcategory"]
        + weights[2] * losses["cooking_style"]
        + weights[3] * losses["calories"]
    )


def move_targets_to_device(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_weights: tuple[float, float, float, float],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {"loss": 0.0}
    count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device)
            targets = move_targets_to_device(targets, device)
            outputs = model(images)
            losses = _losses(outputs, targets)
            total_loss = weighted_total_loss(losses, loss_weights)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            count += batch_size
            totals["loss"] += total_loss.item() * batch_size
            for name, loss_value in losses.items():
                totals[name] = totals.get(name, 0.0) + loss_value.item() * batch_size

    return {name: value / max(count, 1) for name, value in totals.items()}


def train_one_experiment(train_config: config.TrainConfig) -> Path:
    set_seed()
    ensure_dirs()
    train_df, dev_df, test_df = _load_training_frames()
    if train_config.smoke_max_samples is not None:
        cap = train_config.smoke_max_samples
        train_df = train_df.iloc[:cap].copy()
        dev_df = dev_df.iloc[:cap].copy()
    label_maps = build_label_maps(pd.concat([train_df, dev_df, test_df], ignore_index=True))
    save_label_maps(label_maps)

    loaders = make_dataloaders(
        train_df,
        dev_df,
        test_df,
        label_maps,
        augmentation=train_config.augmentation,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )
    device = get_device(train_config.device)
    model = build_model(train_config.model_name, label_maps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    run_name = (
        f"{train_config.model_name}_{train_config.augmentation}_"
        f"w{'-'.join(str(w).replace('.', 'p') for w in train_config.loss_weights)}"
    )
    checkpoint_path = config.CHECKPOINT_DIR / f"{run_name}.pt"
    history: list[dict[str, float | int]] = []
    best_dev_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, train_config.epochs + 1):
        print(f"\nEpoch {epoch}/{train_config.epochs}: {run_name}")
        train_metrics = run_epoch(
            model, loaders["train"], optimizer, device, train_config.loss_weights
        )
        dev_metrics = run_epoch(model, loaders["dev"], None, device, train_config.loss_weights)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(row)
        print(row)

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_maps": label_maps,
                    "train_config": asdict(train_config),
                    "best_dev_loss": best_dev_loss,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                print("Early stopping triggered.")
                break

    history_path = config.METRICS_DIR / f"{run_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    save_json(asdict(train_config), config.METRICS_DIR / f"{run_name}_config.json")
    print(f"Saved training history to {history_path}")
    return checkpoint_path


def run_experiment_grid(base_config: config.TrainConfig) -> list[Path]:
    checkpoints = []
    for augmentation in config.AUGMENTATION_CONFIGS:
        for weights in config.LOSS_WEIGHT_CONFIGS:
            experiment_config = replace(
                base_config,
                augmentation=augmentation,
                loss_weights=weights,
            )
            checkpoints.append(train_one_experiment(experiment_config))
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Train food image models.")
    parser.add_argument("--model", default="multitask_resnet18", choices=["scratch_cnn", "resnet18_subcategory", "multitask_resnet18"])
    parser.add_argument("--augmentation", default="resize", choices=list(config.AUGMENTATION_CONFIGS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke-max-samples",
        type=int,
        default=None,
        help="Use only the first N train/dev rows (for quick CPU smoke).",
    )
    parser.add_argument(
        "--loss-weights",
        nargs=4,
        type=float,
        default=config.LOSS_WEIGHT_CONFIGS[0],
        metavar=("CATEGORY", "SUBCATEGORY", "COOKING_STYLE", "CALORIES"),
    )
    parser.add_argument("--run-experiments", action="store_true")
    args = parser.parse_args()

    require_path(config.FOODNEXTDB_DIR, "FoodNExTDB data is missing.")
    base_config = config.TrainConfig(
        model_name=args.model,
        augmentation=args.augmentation,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss_weights=tuple(args.loss_weights),
        device=args.device,
        smoke_max_samples=args.smoke_max_samples,
    )
    if args.run_experiments:
        run_experiment_grid(base_config)
    else:
        train_one_experiment(base_config)


if __name__ == "__main__":
    main()

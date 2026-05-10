from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from . import config
from .data_loading import FoodImageDataset, get_transforms
from .models import build_model
from .utils import ensure_dirs, get_device, require_path, save_json


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, dict[str, int]], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_maps = checkpoint["label_maps"]
    train_config = checkpoint.get("train_config", {"model_name": "multitask_resnet18"})
    model = build_model(train_config["model_name"], label_maps, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, label_maps, train_config


def _inverse_maps(label_maps: dict[str, dict[str, int]]) -> dict[str, dict[int, str]]:
    return {
        task: {idx: label for label, idx in mapping.items()}
        for task, mapping in label_maps.items()
    }


def collect_predictions(
    model: torch.nn.Module,
    df: pd.DataFrame,
    label_maps: dict[str, dict[str, int]],
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
) -> pd.DataFrame:
    dataset = FoodImageDataset(df, label_maps, get_transforms("resize", train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inverse_maps = _inverse_maps(label_maps)
    rows: list[dict[str, object]] = []
    offset = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            batch_df = df.iloc[offset : offset + images.size(0)].reset_index(drop=True)
            offset += images.size(0)

            if isinstance(outputs, torch.Tensor):
                pred_subcategory = outputs.argmax(dim=1).cpu().numpy()
                for i, pred_idx in enumerate(pred_subcategory):
                    row = batch_df.iloc[i].to_dict()
                    row["pred_subcategory"] = inverse_maps["subcategory"][int(pred_idx)]
                    rows.append(row)
                continue

            pred_indices = {
                task: outputs[task].argmax(dim=1).cpu().numpy()
                for task in ["category", "subcategory", "cooking_style"]
            }
            pred_calories = outputs["calories"].cpu().numpy()
            true_calories = targets["calories"].cpu().numpy()

            for i in range(len(batch_df)):
                row = batch_df.iloc[i].to_dict()
                for task in ["category", "subcategory", "cooking_style"]:
                    row[f"pred_{task}"] = inverse_maps[task][int(pred_indices[task][i])]
                row["pred_calories"] = float(pred_calories[i])
                row["calorie_abs_error"] = abs(float(pred_calories[i]) - float(true_calories[i]))
                rows.append(row)

    return pd.DataFrame(rows)


def classification_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task in ["category", "subcategory", "cooking_style"]:
        pred_col = f"pred_{task}"
        if pred_col not in predictions.columns:
            continue
        metrics[f"{task}_accuracy"] = accuracy_score(predictions[task], predictions[pred_col])
        metrics[f"{task}_macro_f1"] = f1_score(
            predictions[task],
            predictions[pred_col],
            average="macro",
            zero_division=0,
        )
    return metrics


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    if "pred_calories" not in predictions.columns:
        return {}
    mae = mean_absolute_error(predictions["calories"], predictions["pred_calories"])
    mse = mean_squared_error(predictions["calories"], predictions["pred_calories"])
    rmse = float(np.sqrt(mse))
    return {"calorie_mae": float(mae), "calorie_rmse": float(rmse)}


def plot_confusion_matrices(predictions: pd.DataFrame, output_dir: Path = config.FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in ["category", "subcategory", "cooking_style"]:
        pred_col = f"pred_{task}"
        if pred_col not in predictions.columns:
            continue
        labels = sorted(set(predictions[task]) | set(predictions[pred_col]))
        matrix = confusion_matrix(predictions[task], predictions[pred_col], labels=labels)
        plt.figure(figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.35)))
        sns.heatmap(matrix, cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"{task} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        path = output_dir / f"{task}_confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def _plot_examples(rows: pd.DataFrame, title: str, output_path: Path) -> None:
    if rows.empty:
        print(f"No examples available for {title}.")
        return
    rows = rows.head(12)
    cols = 4
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(16, rows_count * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, rows.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        subtitle = (
            f"T: {row.get('subcategory', '')}\n"
            f"P: {row.get('pred_subcategory', '')}\n"
            f"kcal T/P: {row.get('calories', 0):.1f}/{row.get('pred_calories', 0):.1f}"
        )
        ax.set_title(subtitle, fontsize=9)

    for ax in axes[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def error_analysis(predictions: pd.DataFrame, output_dir: Path = config.FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "pred_subcategory" not in predictions.columns:
        return
    correct = predictions[predictions["subcategory"] == predictions["pred_subcategory"]]
    wrong = predictions[predictions["subcategory"] != predictions["pred_subcategory"]]
    high_error = predictions.sort_values("calorie_abs_error", ascending=False) if "calorie_abs_error" in predictions else pd.DataFrame()
    _plot_examples(correct, "Correct subcategory predictions", output_dir / "correct_predictions.png")
    _plot_examples(wrong, "Wrong subcategory predictions", output_dir / "wrong_predictions.png")
    _plot_examples(high_error, "Highest derived-calorie errors", output_dir / "high_calorie_error.png")


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str = "test",
    batch_size: int = config.BATCH_SIZE,
    device_name: str | None = None,
) -> dict[str, float]:
    ensure_dirs()
    require_path(checkpoint_path, "Checkpoint was not found.")
    split_path = config.SPLIT_DIR / f"{split}.csv"
    require_path(split_path, f"{split} split was not found.")
    df = pd.read_csv(split_path)
    device = get_device(device_name)
    model, label_maps, train_config = load_checkpoint(checkpoint_path, device)
    predictions = collect_predictions(model, df, label_maps, device, batch_size=batch_size)

    prediction_path = config.METRICS_DIR / f"{checkpoint_path.stem}_{split}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics = {**classification_metrics(predictions), **regression_metrics(predictions)}
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["model_name"] = train_config.get("model_name", "")

    metrics_path = config.METRICS_DIR / f"{checkpoint_path.stem}_{split}_metrics.json"
    save_json(metrics, metrics_path)
    plot_confusion_matrices(predictions)
    error_analysis(predictions)
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained food image model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.batch_size, args.device)


if __name__ == "__main__":
    main()

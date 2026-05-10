# Multi-Task Food Image Understanding for Dietary Assessment

This project trains PyTorch computer vision models for dietary assessment from food images. Given an image, the main model predicts:

- food `category`
- food `subcategory`
- `cooking_style`
- estimated `calories`

Calories are weak supervision: they are derived by mapping FoodNExTDB subcategories to a nutrition dataset and are not measured ground truth.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── main.ipynb
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── nutrition_mapping.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── data/
    ├── raw/
    └── processed/
```

## Dataset Setup

Download or manually extract FoodNExTDB from:

https://github.com/AI4Food/FoodNExtDB

Place it at:

```text
data/raw/FoodNExTDB/
```

Expected structure:

```text
data/raw/FoodNExTDB/
└── A4F_XXXXX/
    ├── image files
    ├── A4F_XXXXX_labeled_data.csv
    └── A4F_XXXXX_timestamps.csv
```

Each `*_labeled_data.csv` must include:

- `id`
- `category`
- `subcategory`
- `cooking_style`

Download the nutrition dataset:

```bash
curl -L -o data/raw/nutrition/food-nutrition-dataset.zip \
https://www.kaggle.com/api/v1/datasets/download/utsavdey1410/food-nutrition-dataset
```

Then unzip it into:

```text
data/raw/nutrition/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## How To Run

Prepare FoodNExTDB metadata and stratified splits:

```bash
python -m src.data_loading
```

Add weak calorie targets from the nutrition dataset:

```bash
python -m src.nutrition_mapping
```

Train the default multi-task ResNet-18 model:

```bash
python -m src.train --model multitask_resnet18 --epochs 10
```

Run the full augmentation and loss-weight grid:

```bash
python -m src.train --model multitask_resnet18 --epochs 10 --run-experiments
```

Train baselines:

```bash
python -m src.train --model scratch_cnn --epochs 10
python -m src.train --model resnet18_subcategory --epochs 10
```

Evaluate a checkpoint:

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/<checkpoint-name>.pt --split test
```

## Models

- `ScratchCNN`: 4 convolutional blocks with batch norm, ReLU, pooling, dropout, and multi-task heads.
- `resnet18_subcategory`: pretrained ResNet-18 classifier for the subcategory baseline.
- `multitask_resnet18`: pretrained ResNet-18 shared backbone with heads for category, subcategory, cooking style, and calorie regression.

## Experiments

Losses:

- classification: cross entropy
- regression: SmoothL1
- total: weighted sum

Loss weights tested:

- `[1, 1, 1, 0.5]`
- `[1, 2, 1, 1]`
- `[1, 1, 1, 2]`

Augmentation settings:

- baseline resize
- horizontal flip + color jitter
- random crop + rotation

## Outputs

Generated files are written under:

- `data/processed/`: metadata, filtered metadata, split CSVs, nutrition mapping
- `outputs/checkpoints/`: best model checkpoints
- `outputs/metrics/`: histories, predictions, metrics JSON
- `outputs/figures/`: confusion matrices and error-analysis grids

## Known Limitations

- Calorie labels are derived approximations, not ground truth. They do not account for actual portion size, hidden ingredients, plate context, recipe variation, or cooking oil.
- The default `portion_multiplier = 1.0` treats `kcal_per_100g` as the calorie target. This is useful for relative modeling, not reliable dietary logging.
- Nutrition mapping depends on text matching between FoodNExTDB subcategories and the nutrition CSV. Exact, partial, fuzzy, category-average, and global-average fallbacks are used, and unmatched foods are warned about.
- Class imbalance may remain after filtering because FoodNExTDB categories and subcategories are naturally skewed.
- Results are dataset-specific and should not be interpreted as validated clinical dietary assessment.

## Results Summary

Latest results from `outputs/metrics/*_test_metrics.json`:

| Model | Augmentation | Loss Weights | Subcategory Acc | Macro F1 | Calorie MAE | Calorie RMSE |
|---|---|---:|---:|---:|---:|---:|
| ScratchCNN | resize | [1, 1, 1, 0.5] | 0.0178 | 0.0017 | 171.8119 | 204.8656 |
| ResNet-18 baseline | not run | n/a | n/a | n/a | n/a | n/a |
| Multi-task ResNet-18 | resize | [1, 1, 1, 0.5] | 0.4067 | 0.1807 | 152.5093 | 186.2938 |

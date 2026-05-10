from __future__ import annotations

import argparse
import re
import string
import warnings
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except ModuleNotFoundError:
    fuzz = None
    process = None

from . import config
from .data_loading import create_splits, load_metadata, save_metadata
from .utils import ensure_dirs, require_path


NAME_COLUMN_CANDIDATES = [
    "food",
    "food_name",
    "name",
    "item",
    "description",
    "dish",
    "product_name",
]
CALORIE_COLUMN_CANDIDATES = [
    "kcal_per_100g",
    "calories_per_100g",
    "caloric_value",
    "energy_kcal_100g",
    "energy_kcal",
    "calories",
    "kcal",
    "energy",
]


def normalize_food_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def _find_csv_files(nutrition_dir: Path = config.NUTRITION_DIR) -> list[Path]:
    require_path(
        nutrition_dir,
        "Nutrition data was not found. Download and unzip the Kaggle dataset first.",
    )
    csv_files = sorted(nutrition_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {nutrition_dir}.")
    return csv_files


def _infer_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_food_name(col).replace(" ", "_"): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for key, original in normalized.items():
            if candidate in key or key in candidate:
                return original
    return None


def load_nutrition_table(nutrition_dir: Path = config.NUTRITION_DIR) -> pd.DataFrame:
    frames = []
    for csv_path in _find_csv_files(nutrition_dir):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"Skipping unreadable nutrition CSV {csv_path}: {exc}")
            continue

        name_col = _infer_column(list(df.columns), NAME_COLUMN_CANDIDATES)
        calorie_col = _infer_column(list(df.columns), CALORIE_COLUMN_CANDIDATES)
        if name_col is None or calorie_col is None:
            warnings.warn(
                f"Skipping {csv_path}: could not infer food name and calorie columns."
            )
            continue

        slim = df[[name_col, calorie_col]].copy()
        slim.columns = ["food_name", "kcal_per_100g"]
        slim["source_file"] = str(csv_path)
        frames.append(slim)

    if not frames:
        raise ValueError(
            "No usable nutrition CSV was found. Expected columns like food/name and kcal/calories."
        )

    nutrition = pd.concat(frames, ignore_index=True)
    nutrition["kcal_per_100g"] = pd.to_numeric(nutrition["kcal_per_100g"], errors="coerce")
    nutrition = nutrition.dropna(subset=["food_name", "kcal_per_100g"])
    nutrition = nutrition[nutrition["kcal_per_100g"] > 0].copy()
    nutrition["normalized_name"] = nutrition["food_name"].map(normalize_food_name)
    nutrition = nutrition[nutrition["normalized_name"] != ""]
    nutrition = (
        nutrition.groupby("normalized_name", as_index=False)
        .agg(food_name=("food_name", "first"), kcal_per_100g=("kcal_per_100g", "mean"))
        .sort_values("normalized_name")
    )
    print(f"Loaded {len(nutrition)} nutrition entries.")
    return nutrition


def _partial_match(query: str, nutrition: pd.DataFrame) -> tuple[float | None, str | None]:
    for _, row in nutrition.iterrows():
        candidate = row["normalized_name"]
        if query and (query in candidate or candidate in query):
            return float(row["kcal_per_100g"]), str(row["food_name"])
    return None, None


def _fuzzy_match(
    query: str,
    nutrition: pd.DataFrame,
    threshold: int = config.FUZZY_MATCH_THRESHOLD,
) -> tuple[float | None, str | None, int | None]:
    if fuzz is None or process is None:
        raise ImportError("rapidfuzz is required for fuzzy nutrition matching. Run `pip install -r requirements.txt`.")
    choices = nutrition["normalized_name"].tolist()
    result = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
    if result is None:
        return None, None, None
    match, score, idx = result
    if score < threshold:
        return None, None, score
    row = nutrition.iloc[idx]
    return float(row["kcal_per_100g"]), str(row["food_name"]), score


def map_subcategories_to_calories(
    metadata: pd.DataFrame,
    nutrition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_lookup = dict(zip(nutrition["normalized_name"], nutrition["kcal_per_100g"]))
    exact_names = dict(zip(nutrition["normalized_name"], nutrition["food_name"]))
    global_average = float(nutrition["kcal_per_100g"].mean())

    mapping_rows: list[dict[str, object]] = []
    enriched = metadata.copy()

    for subcategory, group in metadata.groupby("subcategory"):
        query = normalize_food_name(subcategory)
        category = str(group["category"].mode().iloc[0])
        kcal = exact_lookup.get(query)
        method = "exact"
        matched_name = exact_names.get(query)
        score: int | None = 100 if kcal is not None else None

        if kcal is None:
            kcal, matched_name = _partial_match(query, nutrition)
            method = "partial" if kcal is not None else method

        if kcal is None:
            kcal, matched_name, score = _fuzzy_match(query, nutrition)
            method = "fuzzy" if kcal is not None else method

        mapping_rows.append(
            {
                "subcategory": subcategory,
                "category": category,
                "normalized_query": query,
                "matched_food_name": matched_name,
                "match_method": method,
                "match_score": score,
                "kcal_per_100g": float(kcal) if kcal is not None else pd.NA,
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    category_means = (
        mapping.dropna(subset=["kcal_per_100g"])
        .groupby("category")["kcal_per_100g"]
        .mean()
        .to_dict()
    )

    for idx, row in mapping[mapping["kcal_per_100g"].isna()].iterrows():
        category = row["category"]
        if category in category_means:
            mapping.loc[idx, "kcal_per_100g"] = float(category_means[category])
            mapping.loc[idx, "match_method"] = "category_average"
            warnings.warn(
                f"No direct nutrition match for subcategory '{row['subcategory']}'. "
                f"Using category average for '{category}'."
            )
        else:
            mapping.loc[idx, "kcal_per_100g"] = global_average
            mapping.loc[idx, "match_method"] = "global_average"
            warnings.warn(
                f"No nutrition match for subcategory '{row['subcategory']}'. "
                f"Using global average {global_average:.1f} kcal/100g."
            )

    enriched = enriched.merge(
        mapping[["subcategory", "kcal_per_100g", "match_method"]],
        on="subcategory",
        how="left",
    )
    # This is a derived approximation, not ground truth.
    enriched["calories"] = enriched["kcal_per_100g"] * config.PORTION_MULTIPLIER
    unmatched = mapping[mapping["match_method"].isin(["category_average", "global_average"])]
    if not unmatched.empty:
        print("Foods requiring fallback nutrition mapping:")
        print(unmatched[["subcategory", "match_method", "kcal_per_100g"]].to_string(index=False))
    return enriched, mapping


def add_calorie_targets(
    metadata_path: Path = config.FILTERED_METADATA_CSV,
    nutrition_dir: Path = config.NUTRITION_DIR,
    output_path: Path = config.MAPPED_METADATA_CSV,
) -> pd.DataFrame:
    ensure_dirs()
    metadata = load_metadata(metadata_path)
    nutrition = load_nutrition_table(nutrition_dir)
    enriched, mapping = map_subcategories_to_calories(metadata, nutrition)
    save_metadata(enriched, output_path)
    create_splits(enriched)
    mapping_path = config.PROCESSED_DATA_DIR / "nutrition_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"Saved nutrition mapping to {mapping_path}")
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Map food image labels to calories.")
    parser.add_argument("--metadata", type=Path, default=config.FILTERED_METADATA_CSV)
    parser.add_argument("--nutrition-dir", type=Path, default=config.NUTRITION_DIR)
    parser.add_argument("--output", type=Path, default=config.MAPPED_METADATA_CSV)
    args = parser.parse_args()
    try:
        add_calorie_targets(args.metadata, args.nutrition_dir, args.output)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()

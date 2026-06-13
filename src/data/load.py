"""
Data loading + feature/target separation for the AI4I 2020 dataset.

Why this module exists separately from training:
- Phase 3 (FastAPI serving) needs the same feature schema at inference time as at
  training time. Centralizing it here means train + serve cannot drift apart.
- A unit test on the schema (tests/test_load.py) is the cheapest defense against
  the most common production ML bug: silent feature-order mismatch.

Schema decisions (defensible in an interview):
- DROP `UDI` and `Product ID`: row identifier and high-cardinality string with no
  predictive signal beyond what `Type` already carries.
- ONE-HOT `Type` (L/M/H — low/medium/high product quality variant). 3 levels,
  no ordinal claim is safe across the dataset, so one-hot beats label encoding.
- EXCLUDE `TWF, HDF, PWF, OSF, RNF` from features. These are *failure-mode* labels
  set only when `Machine failure = 1` — using them as features is target leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Schema ---------------------------------------------------------------

# Columns we drop before modeling (IDs).
ID_COLS = ["UDI", "Product ID"]

# Numeric sensor features kept as-is.
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Categorical feature(s).
CATEGORICAL_FEATURES = ["Type"]

# Target.
TARGET = "Machine failure"

# Leakage trap: sub-failure indicators co-occur with the target — exclude them.
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Map raw CSV names to clean snake_case. XGBoost rejects feature names with
# '[', ']', or '<', and Pydantic models in Phase 3 will be a lot happier with
# names like `torque_nm` than `Torque [Nm]`.
FEATURE_RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

# Clean names — what models and the serving API actually see.
CLEAN_NUMERIC_FEATURES = list(FEATURE_RENAME.values())

DEFAULT_CSV = Path("data/raw/ai4i2020.csv")


@dataclass
class Dataset:
    """A train/test bundle plus the resolved feature-column order."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_raw(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Read the raw CSV. Strips the UTF-8 BOM from the first column header."""
    df = pd.read_csv(csv_path)
    # The UCI file ships with a BOM, so the first header is '﻿UDI'.
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with leakage cols + IDs removed and Type one-hot encoded."""
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    y = df[TARGET].astype(int)

    # One-hot encode Type. drop_first=False keeps all 3 cols so the column set
    # is identical whether or not the data happens to contain all levels.
    X = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=int,
    )
    # Rename raw sensor columns to clean snake_case (XGBoost compatibility +
    # nicer Pydantic schema in Phase 3).
    X = X.rename(columns=FEATURE_RENAME)
    return X, y


def load_dataset(
    csv_path: Path | str = DEFAULT_CSV,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dataset:
    """End-to-end: load CSV → build features → stratified train/test split."""
    df = load_raw(csv_path)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # critical with 3.4% positives
    )

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )

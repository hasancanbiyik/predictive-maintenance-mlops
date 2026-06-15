"""
Smoke tests for the data loader. These exist to catch the silent failures that
otherwise blow up in production: schema drift, leakage cols sneaking back into
X, and broken stratification.

Run: `pytest -q`
"""

from __future__ import annotations

import math

from src.data.load import (
    CLEAN_NUMERIC_FEATURES,
    LEAKAGE_COLS,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    load_dataset,
    load_raw,
)


def test_raw_load_has_expected_shape():
    df = load_raw()
    assert df.shape == (10000, 14), f"Expected (10000, 14), got {df.shape}"
    assert TARGET in df.columns
    for col in NUMERIC_FEATURES:
        assert col in df.columns


def test_features_exclude_leakage_columns():
    df = load_raw()
    X, _ = build_features(df)
    for col in LEAKAGE_COLS:
        assert col not in X.columns, (
            f"Leakage column '{col}' must not be a feature -- it co-occurs with the target."
        )


def test_features_exclude_ids_and_target():
    df = load_raw()
    X, _ = build_features(df)
    for col in ("UDI", "Product ID", TARGET):
        assert col not in X.columns


def test_type_is_one_hot_encoded():
    df = load_raw()
    X, _ = build_features(df)
    # Original Type column gone, replaced by Type_L / Type_M / Type_H
    assert "Type" not in X.columns
    one_hot = [c for c in X.columns if c.startswith("Type_")]
    assert set(one_hot) == {"Type_L", "Type_M", "Type_H"}


def test_stratified_split_preserves_class_ratio():
    ds = load_dataset(random_state=42)
    full_ratio = (ds.y_train.sum() + ds.y_test.sum()) / (len(ds.y_train) + len(ds.y_test))
    train_ratio = ds.y_train.mean()
    test_ratio = ds.y_test.mean()
    # Allow tiny rounding tolerance from stratified split with ~339 positives.
    assert math.isclose(train_ratio, full_ratio, abs_tol=0.005)
    assert math.isclose(test_ratio, full_ratio, abs_tol=0.005)


def test_clean_feature_names_have_no_xgboost_forbidden_chars():
    # XGBoost rejects '[', ']', '<'. This test locks in the rename so the
    # bracket-name regression can never sneak back.
    df = load_raw()
    X, _ = build_features(df)
    for col in X.columns:
        assert not any(ch in col for ch in "[]<"), f"Bad char in feature name: {col!r}"
    for name in CLEAN_NUMERIC_FEATURES:
        assert name in X.columns, f"Expected clean feature {name!r} in X"


def test_feature_names_are_stable_and_ordered():
    ds = load_dataset(random_state=42)
    # Inference-time API must use exactly this column order.
    assert ds.feature_names == list(ds.X_train.columns)
    assert ds.feature_names == list(ds.X_test.columns)

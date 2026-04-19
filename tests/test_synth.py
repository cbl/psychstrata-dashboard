"""Tests for the synthetic-data generator."""

from __future__ import annotations

from models.random_forest_mdd.features import MDD_FEATURES
from models.random_forest_mdd.synth import generate_synthetic_dataset


def test_columns_match_feature_catalog() -> None:
    df, y = generate_synthetic_dataset(n=100, random_state=0)
    assert list(df.columns) == [f.id for f in MDD_FEATURES]
    assert len(y) == len(df) == 100


def test_numeric_values_stay_within_declared_ranges() -> None:
    df, _ = generate_synthetic_dataset(n=500, random_state=1)
    for f in MDD_FEATURES:
        if f.kind == "numeric":
            assert df[f.id].min() >= f.min, f"{f.id} below min"
            assert df[f.id].max() <= f.max, f"{f.id} above max"


def test_categorical_values_are_in_declared_options() -> None:
    df, _ = generate_synthetic_dataset(n=500, random_state=2)
    for f in MDD_FEATURES:
        if f.kind in ("categorical", "ordinal", "binary"):
            allowed = {opt.value for opt in (f.options or ())}
            unseen = set(df[f.id].unique()) - allowed
            assert not unseen, f"{f.id} produced values not in options: {unseen}"


def test_same_seed_produces_identical_output() -> None:
    df_a, y_a = generate_synthetic_dataset(n=200, random_state=42)
    df_b, y_b = generate_synthetic_dataset(n=200, random_state=42)
    assert df_a.equals(df_b)
    assert (y_a == y_b).all()

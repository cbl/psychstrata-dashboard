"""Tests for FeatureEncoder — declarative human-value ↔ model-column boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from models import FeatureEncoder, FeatureSpec, Option


def _num(id_: str, default: float = 0.0) -> FeatureSpec:
    return FeatureSpec(
        id=id_, label=id_.replace("_", " ").title(),
        category="Test", kind="numeric", default=default,
        min=0, max=100, step=1,
    )


def _cat(id_: str, values: list[str], default: str) -> FeatureSpec:
    return FeatureSpec(
        id=id_, label=id_.replace("_", " ").title(),
        category="Test", kind="categorical", default=default,
        options=tuple(Option(label=v, value=v) for v in values),
    )


def _bin(id_: str) -> FeatureSpec:
    return FeatureSpec(
        id=id_, label=id_.replace("_", " ").title(),
        category="Test", kind="binary", default=0,
        options=(Option("No", 0), Option("Yes", 1)),
    )


def _ord(id_: str) -> FeatureSpec:
    return FeatureSpec(
        id=id_, label=id_.replace("_", " ").title(),
        category="Test", kind="ordinal", default=2,
        options=(Option("Low", 0), Option("Mid", 1), Option("High", 2)),
    )


def test_categorical_expands_to_one_hot_in_declaration_order() -> None:
    enc = FeatureEncoder([_cat("sex", ["Female", "Male"], "Female")])
    assert enc.model_feature_names == ["sex_Female", "sex_Male"]


def test_numeric_and_ordinal_and_binary_pass_through_as_single_column() -> None:
    enc = FeatureEncoder([_num("age"), _ord("education"), _bin("self_harm")])
    assert enc.model_feature_names == ["age", "education", "self_harm"]


def test_transform_row_produces_correct_one_hot() -> None:
    enc = FeatureEncoder([_cat("sex", ["Female", "Male"], "Female")])
    out = enc.transform_row({"sex": "Male"})
    assert out.loc[0, "sex_Female"] == 0
    assert out.loc[0, "sex_Male"] == 1


def test_transform_row_keeps_numeric_and_ordinal_values_intact() -> None:
    enc = FeatureEncoder([_num("age"), _ord("education")])
    out = enc.transform_row({"age": 42.0, "education": 3})
    assert out.loc[0, "age"] == 42.0
    assert out.loc[0, "education"] == 3


def test_inverse_feature_name_round_trips_all_kinds() -> None:
    enc = FeatureEncoder([
        _cat("sex", ["Female", "Male"], "Female"),
        _num("age"),
        _bin("self_harm"),
    ])
    assert enc.inverse_feature_name("sex_Female") == "Sex: Female"
    assert enc.inverse_feature_name("age") == "Age"
    assert enc.inverse_feature_name("self_harm") == "Self Harm"


def test_human_value_returns_option_label_not_raw_value() -> None:
    enc = FeatureEncoder([_ord("education"), _bin("self_harm")])
    assert enc.human_value("education", 2) == "High"
    assert enc.human_value("self_harm", 0) == "No"
    assert enc.human_value("self_harm", 1) == "Yes"


def test_transform_df_preserves_column_order_across_features() -> None:
    enc = FeatureEncoder([
        _num("age"),
        _cat("sex", ["Female", "Male"], "Female"),
        _bin("self_harm"),
    ])
    df = pd.DataFrame([{"age": 40, "sex": "Male", "self_harm": 1}])
    out = enc.transform_df(df)
    assert list(out.columns) == ["age", "sex_Female", "sex_Male", "self_harm"]
    assert out.loc[0, "sex_Male"] == 1
    assert out.loc[0, "sex_Female"] == 0

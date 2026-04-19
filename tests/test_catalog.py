"""Tests for the MDD feature catalog — shape and coherence of FEATURES."""

from __future__ import annotations

from models.random_forest_mdd.features import CATEGORY_ORDER, MDD_FEATURES


def test_catalog_has_no_duplicate_ids() -> None:
    ids = [f.id for f in MDD_FEATURES]
    assert len(ids) == len(set(ids)), f"duplicate feature ids: {[i for i in ids if ids.count(i) > 1]}"


def test_every_default_is_in_options_for_categorical_ordinal_binary() -> None:
    for f in MDD_FEATURES:
        if f.kind in ("categorical", "ordinal", "binary"):
            assert f.options is not None, f"{f.id} ({f.kind}) has no options"
            values = {opt.value for opt in f.options}
            assert f.default in values, f"{f.id} default {f.default!r} not in {sorted(map(str, values))}"


def test_every_numeric_has_min_max_step() -> None:
    for f in MDD_FEATURES:
        if f.kind == "numeric":
            assert f.min is not None, f"{f.id} missing min"
            assert f.max is not None, f"{f.id} missing max"
            assert f.step is not None, f"{f.id} missing step"
            assert f.min < f.max, f"{f.id} min ≥ max"


def test_every_non_numeric_has_options() -> None:
    for f in MDD_FEATURES:
        if f.kind != "numeric":
            assert f.options and len(f.options) >= 2, f"{f.id} must have ≥2 options"


def test_all_categories_are_declared_in_category_order() -> None:
    declared = set(CATEGORY_ORDER)
    used = {f.category for f in MDD_FEATURES}
    missing = used - declared
    assert not missing, f"categories not in CATEGORY_ORDER: {missing}"

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd


FeatureKind = Literal["numeric", "categorical", "ordinal", "binary"]


@dataclass(frozen=True)
class Option:
    label: str
    value: Any


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    label: str
    category: str
    kind: FeatureKind
    default: Any
    description: str | None = None
    options: tuple[Option, ...] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None


class FeatureEncoder:
    """Human values ↔ model-input columns. Declarative: no .fit required."""

    def __init__(self, features: list[FeatureSpec]):
        self._features = list(features)
        self._by_id = {f.id: f for f in self._features}
        self._model_feature_names = self._expand_columns()

    @property
    def features(self) -> list[FeatureSpec]:
        return self._features

    @property
    def model_feature_names(self) -> list[str]:
        return list(self._model_feature_names)

    def transform_row(self, raw: dict[str, Any]) -> pd.DataFrame:
        return self.transform_df(pd.DataFrame([raw]))

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, Any] = {}
        for f in self._features:
            values = df[f.id]
            if f.kind == "categorical":
                for opt in f.options or ():
                    cols[f"{f.id}_{opt.value}"] = (values == opt.value).astype(int).values
            else:
                cols[f.id] = values.values
        return pd.DataFrame(cols, columns=self._model_feature_names)

    def inverse_feature_name(self, model_col: str) -> str:
        for f in self._features:
            if f.kind == "categorical":
                for opt in f.options or ():
                    if model_col == f"{f.id}_{opt.value}":
                        return f"{f.label}: {opt.label}"
            elif model_col == f.id:
                return f.label
        return model_col

    def human_value(self, feature_id: str, value: Any) -> str:
        f = self._by_id[feature_id]
        if f.options:
            for opt in f.options:
                if opt.value == value:
                    return opt.label
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _expand_columns(self) -> list[str]:
        names: list[str] = []
        for f in self._features:
            if f.kind == "categorical":
                names.extend(f"{f.id}_{opt.value}" for opt in f.options or ())
            else:
                names.append(f.id)
        return names


@dataclass
class Contributor:
    feature_id: str
    human_label: str
    selected_value: Any
    shap_value: float


@dataclass
class Prediction:
    label: str
    probability: float | None = None
    cause_curves: dict[str, dict[float, float]] | None = None
    conformal_set: list[str] | None = None


@dataclass
class Explanation:
    shap_values: dict[str, float]
    top_positive: list[Contributor] = field(default_factory=list)
    top_negative: list[Contributor] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureEvidence:
    association: str
    pmids: tuple[str, ...] = ()
    note: str = ""


class TRModel(ABC):
    name: str
    diagnosis: str
    features: list[FeatureSpec]
    encoder: FeatureEncoder
    evidence: dict[str, FeatureEvidence] = {}

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Path) -> "TRModel": ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def predict(self, raw: dict[str, Any]) -> Prediction: ...

    @abstractmethod
    def explain(self, raw: dict[str, Any]) -> Explanation: ...

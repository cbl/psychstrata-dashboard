from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from mapie.classification import MapieClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from ..base import (
    Contributor,
    Explanation,
    FeatureEncoder,
    FeatureSpec,
    Prediction,
    TRModel,
)
from .evidence import FEATURE_EVIDENCE
from .features import MDD_FEATURES
from .synth import generate_synthetic_dataset

ARTIFACTS_FILENAME = "artifacts.pkl"


class RandomForestMDD(TRModel):
    name = "random_forest_mdd"
    diagnosis = "MDD"

    def __init__(self):
        self.features: list[FeatureSpec] = MDD_FEATURES
        self.encoder = FeatureEncoder(MDD_FEATURES)
        self.evidence = FEATURE_EVIDENCE
        self.auc: float = 0.0
        self._rf: RandomForestClassifier | None = None
        self._mapie: MapieClassifier | None = None
        self._shap: shap.TreeExplainer | None = None
        self._tsne: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None
        self._y: np.ndarray | None = None

    @classmethod
    def train(cls, n_samples: int = 2500, random_state: int = 42) -> "RandomForestMDD":
        model = cls()
        df_human, y = generate_synthetic_dataset(n=n_samples, random_state=random_state)
        X = model.encoder.transform_df(df_human)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y,
        )
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train,
        )

        rf = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3,
            random_state=random_state, n_jobs=-1,
        )
        rf.fit(X_fit, y_fit)

        mapie = MapieClassifier(estimator=rf, method="score", cv="prefit")
        mapie.fit(X_calib, y_calib)

        background = X_fit.sample(min(200, len(X_fit)), random_state=random_state)
        shap_explainer = shap.TreeExplainer(
            rf, data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )

        tsne = TSNE(
            n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=random_state,
        ).fit_transform(X.values)
        nn = NearestNeighbors(n_neighbors=15, metric="euclidean").fit(X.values)

        model._rf = rf
        model._mapie = mapie
        model._shap = shap_explainer
        model._tsne = tsne
        model._nn = nn
        model.auc = float(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
        model._y = y.values
        return model

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "rf": self._rf, "mapie": self._mapie, "shap": self._shap,
            "tsne": self._tsne, "nn": self._nn, "auc": self.auc, "y": self._y,
        }
        with open(path / ARTIFACTS_FILENAME, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def from_pretrained(cls, path: Path) -> "RandomForestMDD":
        with open(Path(path) / ARTIFACTS_FILENAME, "rb") as f:
            payload = pickle.load(f)
        model = cls()
        model._rf = payload["rf"]
        model._mapie = payload["mapie"]
        model._shap = payload["shap"]
        model._tsne = payload["tsne"]
        model._nn = payload["nn"]
        model.auc = float(payload["auc"])
        model._y = payload["y"]
        return model

    def predict(self, raw: dict[str, Any]) -> Prediction:
        X_row = self.encoder.transform_row(raw)
        prob = float(self._rf.predict_proba(X_row)[0, 1])
        label, conformal_set = self._conformal(X_row, ci_level=95)
        return Prediction(label=label, probability=prob, conformal_set=conformal_set)

    def explain(self, raw: dict[str, Any]) -> Explanation:
        X_row = self.encoder.transform_row(raw)
        shap_vals = self._shap_for_row(X_row)
        shap_dict = dict(zip(self.encoder.model_feature_names, shap_vals))
        positive, negative = self._rank_contributors(shap_dict, raw)
        return Explanation(shap_values=shap_dict, top_positive=positive, top_negative=negative)

    def tsne_position(self, raw: dict[str, Any], k: int = 15) -> tuple[float, float]:
        X_row = self.encoder.transform_row(raw)
        distances, indices = self._nn.kneighbors(X_row.values, n_neighbors=k)
        d, idx = distances[0], indices[0]
        if np.all(d == 0):
            pos = self._tsne[idx[0]]
        else:
            w = 1.0 / (d + 1e-8)
            w = w / w.sum()
            pos = (self._tsne[idx] * w[:, None]).sum(axis=0)
        return float(pos[0]), float(pos[1])

    @property
    def tsne_embedding(self) -> np.ndarray:
        return self._tsne

    @property
    def y_labels(self) -> np.ndarray:
        return self._y

    def _shap_for_row(self, X_row: pd.DataFrame) -> np.ndarray:
        sv = self._shap.shap_values(X_row)
        arr = np.asarray(sv[-1] if isinstance(sv, list) else sv, dtype=float)
        if arr.ndim == 3:
            return arr[0, :, -1]
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return np.squeeze(arr)

    def _conformal(self, X_row: pd.DataFrame, ci_level: int) -> tuple[str, list[str]]:
        alpha = 1.0 - (ci_level / 100.0)
        _, y_ps = self._mapie.predict(X_row, alpha=alpha)
        if y_ps.ndim == 3:
            y_ps = y_ps[:, :, 0]
        included = y_ps[0].astype(bool)
        classes = list(self._rf.classes_)
        set_labels = [
            "Resistant" if classes[i] == 1 else "Responsive"
            for i in range(len(included)) if included[i]
        ]
        if len(set_labels) != 1:
            return "Uncertain", set_labels
        return set_labels[0], set_labels

    def _rank_contributors(
        self, shap_dict: dict[str, float], raw: dict[str, Any], k: int = 3,
    ) -> tuple[list[Contributor], list[Contributor]]:
        ranked = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        positive: list[Contributor] = []
        negative: list[Contributor] = []
        for model_col, sv in ranked:
            if len(positive) >= k and len(negative) >= k:
                break
            feature_id = self._feature_id_for(model_col)
            c = Contributor(
                feature_id=feature_id,
                human_label=self.encoder.inverse_feature_name(model_col),
                selected_value=self.encoder.human_value(feature_id, raw.get(feature_id)),
                shap_value=float(sv),
            )
            if sv > 0 and len(positive) < k:
                positive.append(c)
            elif sv < 0 and len(negative) < k:
                negative.append(c)
        return positive, negative

    def _feature_id_for(self, model_col: str) -> str:
        for f in self.features:
            if f.kind == "categorical" and model_col.startswith(f"{f.id}_"):
                return f.id
            if model_col == f.id:
                return f.id
        return model_col

"""DeepHit competing-risks wrapper adapted to the dashboard TRModel interface.

Loads a fitted TransformPipeline + DeepHit state_dict if present in the
artifacts directory. Otherwise fits a placeholder pipeline on synthetic
MDD data and trains a fresh network so the UI stays interactive.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    import torch
    import torch.nn as nn
    from psychstrata.data.synthetic import CohortConfig, generate_cohort
    from psychstrata.data.transforms import TransformConfig, TransformPipeline
    from psychstrata.nn.models.deephit import DeepHit, DeepHitConfig, deephit_loss

    _DEEPHIT_AVAILABLE = True
    _DEEPHIT_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:
    _DEEPHIT_AVAILABLE = False
    _DEEPHIT_IMPORT_ERROR = _exc

from ..base import (
    Contributor,
    Explanation,
    FeatureEncoder,
    FeatureSpec,
    Prediction,
    TRModel,
)
from ..random_forest_mdd.features import MDD_FEATURES
from .adapter import ui_to_pipeline_df
from .evidence import FEATURE_EVIDENCE


def _require_deephit() -> None:
    if not _DEEPHIT_AVAILABLE:
        raise ImportError(
            "DeepHit requires the psychstrata package (plus torch + shap).\n"
            "Install it with: pip install psychstrata torch shap\n"
            f"Original import error: {_DEEPHIT_IMPORT_ERROR}"
        )

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "deephit_mdd"
DEFAULT_N_SYNTH = 2000
DEFAULT_SEED = 42
DEFAULT_PLACEHOLDER_EPOCHS = 30
TR_CAUSE_INDEX = 0
HORIZON_YEARS = 5.0

CAUSE_LABELS: list[str] = ["Treatment resistance", "Death", "Discontinuation"]
SHAP_BACKGROUND_N = 50
SHAP_TOP_K = 3


class DeepHitMDD(TRModel):
    name = "deephit_mdd"
    diagnosis = "MDD"

    def __init__(self):
        self.features = MDD_FEATURES
        self.encoder = FeatureEncoder(MDD_FEATURES)
        self.evidence = FEATURE_EVIDENCE
        self.auc: float = 0.0
        self.horizon_years: float = HORIZON_YEARS
        self._pipeline: TransformPipeline | None = None
        self._model: DeepHit | None = None
        self._time_bins: np.ndarray | None = None
        self._shap_explainer: shap.GradientExplainer | None = None
        self._pipeline_feature_names: list[str] = []

    @classmethod
    def build(cls, artifacts_dir: Path | None = None) -> "DeepHitMDD":
        _require_deephit()
        artifacts_dir = Path(
            artifacts_dir
            or os.environ.get("DEEPHIT_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
        )

        pipeline = cls._load_or_fit_pipeline(artifacts_dir)
        train_df = _synthetic_mdd_cohort()
        X, _, _, T_discrete = pipeline.transform(train_df)
        E = train_df["E_tr"].values.astype(np.int64)

        n_causes = max(int(E.max()), 1)
        config = DeepHitConfig(
            in_features=X.shape[1],
            n_causes=n_causes,
            n_time_bins=pipeline.n_time_bins,
            shared_hidden=[128, 128],
            cause_hidden=[64],
            dropout=0.1,
        )
        network = DeepHit(config)

        if not cls._try_load_state(network, artifacts_dir, config):
            logger.info(
                "DeepHit: no compatible checkpoint — training %d-epoch placeholder on synthetic data",
                DEFAULT_PLACEHOLDER_EPOCHS,
            )
            _quick_train(network, X, T_discrete, E, epochs=DEFAULT_PLACEHOLDER_EPOCHS)

        instance = cls()
        instance._pipeline = pipeline
        instance._model = network
        instance._time_bins = pipeline.get_time_bins()
        instance._pipeline_feature_names = pipeline.get_feature_names()
        instance._init_shap(X)
        return instance

    @classmethod
    def from_pretrained(cls, path: Path) -> "DeepHitMDD":
        return cls.build(artifacts_dir=Path(path))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._pipeline is not None and self._model is not None
        self._pipeline.save(path / "pipeline.pkl")
        torch.save(self._model.state_dict(), path / "model_state.pt")
        config_dict = {
            "type": "deephit",
            "in_features": int(self._model.config.in_features),
            "n_causes": int(self._model.config.n_causes),
            "n_time_bins": int(self._model.config.n_time_bins),
            "shared_hidden": list(self._model.config.shared_hidden),
            "cause_hidden": list(self._model.config.cause_hidden),
            "dropout": float(self._model.config.dropout),
        }
        (path / "model_config.json").write_text(json.dumps(config_dict, indent=2))

    def predict(self, raw: dict[str, Any]) -> Prediction:
        cif_all, bin_centers = self._cif_all_causes(raw)
        cause_curves: dict[str, dict[float, float]] = {}
        for i in range(cif_all.shape[0]):
            label = CAUSE_LABELS[i] if i < len(CAUSE_LABELS) else f"Cause {i + 1}"
            cause_curves[label] = {float(t): float(c) for t, c in zip(bin_centers, cif_all[i])}

        tr_curve = cif_all[min(TR_CAUSE_INDEX, cif_all.shape[0] - 1)]
        horizon_prob = float(_cif_at_time(tr_curve, bin_centers, self.horizon_years))
        label = "Resistant" if horizon_prob >= 0.5 else "Responsive"
        return Prediction(
            label=label,
            probability=horizon_prob,
            cause_curves=cause_curves,
            conformal_set=[label],
        )

    def explain(self, raw: dict[str, Any]) -> Explanation:
        if self._shap_explainer is None:
            return Explanation(shap_values={}, top_positive=[], top_negative=[])

        df = ui_to_pipeline_df(raw)
        X, _, _, _ = self._pipeline.transform(df)
        x_tensor = torch.from_numpy(X).float()
        shap_raw = self._shap_explainer.shap_values(x_tensor)
        values = np.asarray(shap_raw).reshape(-1)[: len(self._pipeline_feature_names)]
        shap_dict = dict(zip(self._pipeline_feature_names, values.tolist()))

        positive, negative = self._rank_contributors(shap_dict, raw)
        return Explanation(shap_values=shap_dict, top_positive=positive, top_negative=negative)

    def tsne_position(self, raw: dict[str, Any]) -> tuple[float, float]:
        return 0.0, 0.0

    @property
    def tsne_embedding(self) -> np.ndarray:
        return np.zeros((0, 2))

    @property
    def y_labels(self) -> np.ndarray:
        return np.zeros((0,), dtype=int)

    @staticmethod
    def _load_or_fit_pipeline(artifacts_dir: Path) -> TransformPipeline:
        path = artifacts_dir / "pipeline.pkl"
        if path.exists():
            logger.info("DeepHit: loaded pipeline from %s", path)
            return TransformPipeline.load(path)

        logger.info("DeepHit: fitting placeholder pipeline on synthetic data")
        config = TransformConfig(
            with_prs=True, with_pcs=True, with_demographics=True,
            diagnosis_filter="MDD", seed=DEFAULT_SEED,
        )
        pipeline = TransformPipeline(config).fit(_synthetic_mdd_cohort())
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save(path)
        return pipeline

    @staticmethod
    def _try_load_state(network: DeepHit, artifacts_dir: Path, config: DeepHitConfig) -> bool:
        state_path = artifacts_dir / "model_state.pt"
        config_path = artifacts_dir / "model_config.json"
        if not (state_path.exists() and config_path.exists()):
            return False

        saved = json.loads(config_path.read_text())
        if (
            saved.get("type") != "deephit"
            or saved.get("in_features") != config.in_features
            or saved.get("n_causes") != config.n_causes
            or saved.get("n_time_bins") != config.n_time_bins
        ):
            logger.warning(
                "DeepHit: checkpoint shape mismatch — expected (in_features=%d, n_causes=%d, "
                "n_time_bins=%d) got %s; falling back to placeholder training",
                config.in_features, config.n_causes, config.n_time_bins, saved,
            )
            return False

        network.load_state_dict(torch.load(state_path, map_location="cpu"))
        logger.info("DeepHit: loaded state_dict from %s", state_path)
        return True

    def _init_shap(self, X_train: np.ndarray) -> None:
        try:
            bin_centers = 0.5 * (self._time_bins[:-1] + self._time_bins[1:])
            horizon_bin = int(np.argmin(np.abs(bin_centers - self.horizon_years)))
            wrapper = _TRHorizonWrapper(self._model, TR_CAUSE_INDEX, horizon_bin)
            rng = np.random.default_rng(DEFAULT_SEED)
            n_bg = min(SHAP_BACKGROUND_N, len(X_train))
            bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
            background = torch.from_numpy(X_train[bg_idx]).float()
            self._shap_explainer = shap.GradientExplainer(wrapper, background)
        except Exception as exc:
            logger.warning("DeepHit SHAP init failed (%s) — explanations disabled", exc)
            self._shap_explainer = None

    def _rank_contributors(
        self, shap_dict: dict[str, float], raw: dict[str, Any],
    ) -> tuple[list[Contributor], list[Contributor]]:
        ranked = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        positive: list[Contributor] = []
        negative: list[Contributor] = []
        for col, sv in ranked:
            if len(positive) >= SHAP_TOP_K and len(negative) >= SHAP_TOP_K:
                break
            human_label, feature_id = _humanize_pipeline_col(col, self.features)
            selected = _selected_value_for_col(col, raw, self.encoder)
            c = Contributor(
                feature_id=feature_id,
                human_label=human_label,
                selected_value=selected,
                shap_value=float(sv),
            )
            if sv > 0 and len(positive) < SHAP_TOP_K:
                positive.append(c)
            elif sv < 0 and len(negative) < SHAP_TOP_K:
                negative.append(c)
        return positive, negative

    def _cif_all_causes(self, raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        assert self._pipeline is not None and self._model is not None and self._time_bins is not None
        df = ui_to_pipeline_df(raw)
        X, _, _, _ = self._pipeline.transform(df)
        with torch.no_grad():
            cif = self._model.predict_cif(torch.from_numpy(X).float()).cpu().numpy()[0]
        bin_centers = 0.5 * (self._time_bins[:-1] + self._time_bins[1:])
        return cif, bin_centers


def _synthetic_mdd_cohort(seed: int = DEFAULT_SEED, n_mdd: int = DEFAULT_N_SYNTH) -> pd.DataFrame:
    return generate_cohort(CohortConfig(n_mdd=n_mdd, n_scz=0, n_bd=0, seed=seed))


def _quick_train(
    network: DeepHit,
    X: np.ndarray,
    T_discrete: np.ndarray,
    E: np.ndarray,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> None:
    X_t = torch.from_numpy(X).float()
    T_t = torch.from_numpy(T_discrete).long()
    E_t = torch.from_numpy(E).long()
    n = X_t.shape[0]
    opt = torch.optim.Adam(network.parameters(), lr=lr)
    network.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pmf = network(X_t[idx])
            loss = deephit_loss(pmf, T_t[idx], E_t[idx])
            loss.backward()
            opt.step()
    network.eval()


def _cif_at_time(cif_curve: np.ndarray, bin_centers: np.ndarray, t: float) -> float:
    if len(bin_centers) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(bin_centers - t)))
    return float(cif_curve[idx])


if _DEEPHIT_AVAILABLE:

    class _TRHorizonWrapper(nn.Module):
        """Scalar wrapper around DeepHit: returns P(cause @ horizon bin) per sample."""

        def __init__(self, network: DeepHit, cause_index: int, horizon_bin: int):
            super().__init__()
            self.network = network
            self.cause_index = cause_index
            self.horizon_bin = horizon_bin

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pmf = self.network(x)
            cif = pmf.cumsum(dim=-1)
            return cif[:, self.cause_index, self.horizon_bin].unsqueeze(-1)


def _humanize_pipeline_col(col: str, features: list[FeatureSpec]) -> tuple[str, str]:
    for f in features:
        if col == f.id:
            return f.label, f.id
    for f in features:
        prefix = f"{f.id}_"
        if col.startswith(prefix):
            return f"{f.label}: {col[len(prefix):]}", f.id
    return col, col


def _selected_value_for_col(col: str, raw: dict[str, Any], encoder: FeatureEncoder) -> Any:
    if col in raw:
        return encoder.human_value(col, raw[col])
    for f in encoder.features:
        prefix = f"{f.id}_"
        if col.startswith(prefix) and f.kind in ("categorical", "ordinal", "binary"):
            level = col[len(prefix):]
            raw_value = raw.get(f.id)
            return encoder.human_value(f.id, raw_value) if str(raw_value) == level else "—"
    return "—"

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
import torch
from psychstrata.data.synthetic import CohortConfig, generate_cohort
from psychstrata.data.transforms import TransformConfig, TransformPipeline
from psychstrata.nn.models.deephit import DeepHit, DeepHitConfig, deephit_loss

from ..base import (
    Explanation,
    FeatureEncoder,
    Prediction,
    TRModel,
)
from ..random_forest_mdd.features import MDD_FEATURES
from .adapter import ui_to_pipeline_df
from .evidence import FEATURE_EVIDENCE

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("/home/lcb/scai/psychstrata2/models")
DEFAULT_N_SYNTH = 2000
DEFAULT_SEED = 42
DEFAULT_PLACEHOLDER_EPOCHS = 30
TR_CAUSE_INDEX = 0
HORIZON_YEARS = 5.0

CAUSE_LABELS: list[str] = ["Treatment resistance", "Death", "Discontinuation"]


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

    @classmethod
    def build(cls, artifacts_dir: Path | None = None) -> "DeepHitMDD":
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
        return Explanation(shap_values={}, top_positive=[], top_negative=[])

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

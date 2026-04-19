from __future__ import annotations

import importlib
import os

from .base import TRModel

DEFAULT_MODEL = "random_forest_mdd"


def load_model(name: str | None = None) -> TRModel:
    name = name or os.environ.get("PSYCHSTRATA_MODEL", DEFAULT_MODEL)
    module = importlib.import_module(f"models.{name}")
    return module.build_model()

"""Bridge: 46-feature UI dict → 76-column DataFrame the TransformPipeline expects.

The pipeline was trained on the full Psych-STRATA primary catalog (76 features).
The dashboard UI exposes 46 of those (no genetics). Missing genetics columns are
filled with NaN for continuous features (imputer handles them) and a sensible
default for categorical pharmacogenomic features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..random_forest_mdd.features import MDD_FEATURES


PRS_COLS: list[str] = ["prs_mdd", "prs_scz", "prs_bd", "prs_adhd"]
POPULATION_PC_COLS: list[str] = [f"genetic_pc_{i}" for i in range(1, 21)]
CYP_COLS: list[str] = ["cyp2d6_phenotype", "cyp2c19_phenotype", "cyp2b6_phenotype"]
HLA_COLS: list[str] = ["hla_dqb1_126q", "hla_b_158t", "hla_drb1_0402"]

SEX_TO_BINARY: dict[str, int] = {"Female": 0, "Male": 1}
CYP_DEFAULT = "NM"


def ui_to_pipeline_df(raw: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for f in MDD_FEATURES:
        value = raw[f.id]
        row[f.id] = SEX_TO_BINARY[value] if f.id == "sex" else value

    for col in PRS_COLS + POPULATION_PC_COLS:
        row[col] = np.nan
    for col in CYP_COLS:
        row[col] = CYP_DEFAULT
    for col in HLA_COLS:
        row[col] = 0

    row["T_tr"] = 0.0
    row["E_tr"] = 0
    row["diagnosis"] = "MDD"
    return pd.DataFrame([row])

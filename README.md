# Psych-STRATA Treatment Resistance Dashboard

Interactive web demo for predicting treatment resistance in Major Depressive Disorder. Built as the consortium-facing front-end for the Psych-STRATA WP2 prognostic modelling work.

**This demo uses fully synthetic data. It is not a medical device and must not be used for clinical decisions.**

## What's in it

- Pluggable model layer (`models/`) — swap between a RandomForest binary classifier and a DeepHit competing-risks model with one env var
- 46-feature MDD input form grouped by clinical category (demographics, severity, comorbidities, treatment history, biomarkers, family history, …)
- RandomForest: conformal-prediction uncertainty (MAPIE), SHAP feature attributions with human labels, t-SNE population map
- DeepHit: Cumulative-Incidence plot for three competing events (TR / death / discontinuation) plus event-free remainder
- LLM-generated plain-language explanation grounded in SHAP output + curated literature evidence
- REST API (`/api/predict`, `/api/features`, `/api/health`)

## Repository structure

```
psychstrata-dashboard/
├── models/
│   ├── base.py                     FeatureSpec, FeatureEncoder, TRModel interface
│   ├── registry.py                 load_model(name)
│   ├── random_forest_mdd/          Binary classifier, 46 features
│   │   ├── features.py
│   │   ├── synth.py
│   │   ├── model.py
│   │   └── evidence.py
│   └── deephit_mdd/                Competing-risks survival, same 46 UI features
│       ├── features.py             (re-exports MDD_FEATURES)
│       ├── adapter.py              UI dict → 76-col DataFrame for pipeline
│       ├── model.py                DeepHitMDD(TRModel)
│       └── evidence.py
├── app.py                          Dash app entry
├── components.py                   UI widgets
├── callbacks.py                    Wires UI → model.predict / .explain
├── visualization.py                Plotly figures (gauge, CIF, SHAP, t-SNE)
├── api.py                          Flask REST
├── llm_summary.py                  OpenAI-backed explanation
├── config.py                       UI style constants
├── tests/                          pytest suite
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile
```

## Model selection

| Model | Activates with | Output shown in UI |
|---|---|---|
| RandomForest (default) | `PSYCHSTRATA_MODEL=random_forest_mdd` or unset | Probability gauge + conformal badge + SHAP bars + t-SNE |
| DeepHit | `PSYCHSTRATA_MODEL=deephit_mdd` | Probability gauge for P(TR @ 5y) + CIF curves for all 3 competing events + SHAP bars |

### DeepHit artifacts

`DeepHitMDD.build()` looks for model weights + fitted preprocessing pipeline at `DEEPHIT_ARTIFACTS_DIR` (default: `./deephit/`). It expects:

```
deephit/
├── pipeline.pkl          fitted psychstrata TransformPipeline
├── model_state.pt        torch state_dict
└── model_config.json     DeepHit architecture + in_features/n_causes/n_time_bins
```

If the directory is empty or the checkpoint shape mismatches the fitted pipeline, `build()` falls back to fitting a placeholder pipeline on synthetic MDD data and training a fresh 30-epoch network. Subsequent runs re-train (no cache yet — see `TODO` below).

To point at real artifacts:

```bash
DEEPHIT_ARTIFACTS_DIR=/path/to/real/artifacts \
PSYCHSTRATA_MODEL=deephit_mdd \
.venv/bin/python app.py
```

## Adding a new model

1. Create `models/<your_model>/` with:
   - `features.py` — list of `FeatureSpec`
   - `model.py` — subclass of `TRModel` implementing `from_pretrained`, `save`, `predict`, `explain`
   - `__init__.py` — expose `build_model()`
2. Select it at runtime: `PSYCHSTRATA_MODEL=<your_model> python app.py`

No Dash code needs to change — the UI is driven entirely by `model.features`.

## Running locally

**Standalone (RandomForest only):**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                   # http://localhost:8050
```

**With DeepHit:** DeepHit depends on the external `psychstrata` package (not on PyPI). Clone it separately and install as editable:

```bash
git clone <psychstrata-repo-url> /path/to/psychstrata
.venv/bin/pip install -e /path/to/psychstrata
PSYCHSTRATA_MODEL=deephit_mdd .venv/bin/python app.py
```

If `psychstrata` isn't importable, DeepHit surfaces a clear `ImportError` at build time; the dashboard still works with the default RandomForest model.

Production-like:

```bash
.venv/bin/gunicorn app:server -b 0.0.0.0:8050
```

## Docker

```bash
docker build -t psychstrata-dashboard .
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" -p 8050:8050 psychstrata-dashboard
```

The image ships with RandomForest only. To run DeepHit in a container you'd extend the Dockerfile to also `pip install` psychstrata from its source (not included here — depends on where that package is hosted).

Mount real DeepHit artifacts from the host:

```bash
docker run \
  -e PSYCHSTRATA_MODEL=deephit_mdd \
  -e DEEPHIT_ARTIFACTS_DIR=/artifacts \
  -v /host/path/to/artifacts:/artifacts \
  -p 8050:8050 psychstrata-dashboard
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Feature set

46 features, grouped into 12 clinical categories, modelled on the Psych-STRATA WP2 primary feature catalog (genetics excluded for this dashboard). Feature definitions are declared in `models/random_forest_mdd/features.py`.

## Evidence for key predictors

Literature evidence (PMIDs) in `models/<model>/evidence.py` is added incrementally — each entry verified against PubMed before inclusion.

## License

See [LICENSE](LICENSE).

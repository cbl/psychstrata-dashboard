# Psych-STRATA Treatment Resistance Dashboard

Interactive web demo for predicting treatment resistance in Major Depressive Disorder. Built as the consortium-facing front-end for the Psych-STRATA WP2 prognostic modelling work.

**This demo uses fully synthetic data. It is not a medical device and must not be used for clinical decisions.**

## What's in it

- Pluggable model layer (`models/`) — swap RandomForest for DeepHit / HACSurv / Cox by pointing at a different model folder
- 46-feature MDD-specific input form, grouped by clinical category (demographics, severity, comorbidities, treatment history, biomarkers, family history, …)
- RandomForest classifier with conformal-prediction uncertainty (MAPIE)
- SHAP feature attributions with human-readable labels
- t-SNE population map
- LLM-generated plain-language explanation grounded in SHAP output plus curated literature evidence
- REST API (`/api/predict`, `/api/features`, `/api/health`)

## Repository structure

```
psychstrata-dashboard/
├── models/
│   ├── base.py                     FeatureSpec, FeatureEncoder, TRModel interface
│   ├── registry.py                 load_model(name)
│   └── random_forest_mdd/
│       ├── features.py             46 MDD FeatureSpec
│       ├── synth.py                Synthetic-data generator
│       ├── model.py                RandomForestMDD(TRModel)
│       └── evidence.py             Literature evidence per feature
├── app.py                          Dash app entry
├── components.py                   UI widgets (render FeatureSpec)
├── callbacks.py                    Wires UI → model.predict / .explain
├── visualization.py                Plotly figures
├── api.py                          Flask REST
├── llm_summary.py                  LLM explanation
├── config.py                       UI style constants
├── requirements.txt
└── Dockerfile
```

## Adding a new model

1. Create `models/<your_model>/` with:
   - `features.py` — list of `FeatureSpec`
   - `model.py` — subclass of `TRModel` implementing `train`, `from_pretrained`, `save`, `predict`, `explain`
   - `__init__.py` — expose `build_model()`
2. Select it at runtime: `PSYCHSTRATA_MODEL=<your_model> python app.py`

No Dash code needs to change — the UI is driven by `model.features`.

## Running

```bash
pip install -r requirements.txt
python app.py                       # http://localhost:8050
```

Production-like:

```bash
gunicorn app:server -b 0.0.0.0:8050
```

Docker:

```bash
docker build -t psychstrata-dashboard .
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" -p 8050:8050 psychstrata-dashboard
```

The LLM explanation uses `OPENAI_API_KEY` from the environment. The prompt is constrained to live SHAP output plus the literature snippets in `models/random_forest_mdd/evidence.py`.

## Feature set

The 46 features are the MDD subset of the Psych-STRATA primary feature catalog (`code/docs/features.md` in the sibling `code/` repo), excluding the 30 genetics features. Grouped into 12 categories. Feature definitions and literature references live in that file.

## Evidence for key predictors

The feature catalog is populated in `models/random_forest_mdd/features.py`. Literature evidence (PMIDs) in `models/random_forest_mdd/evidence.py` is added incrementally — each entry is verified against PubMed before inclusion.

Full literature documentation for the underlying predictors lives in the sibling `code/docs/features.md`.

## License

See [LICENSE](LICENSE).

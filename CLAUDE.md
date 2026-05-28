# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full pipeline (regenerates all artifacts from raw CSV)
bash run_all.sh

# Individual steps
python -m src.eda                               # EDA charts -> reports/figures/
python -m src.preprocessing                     # train/test split -> data/processed/
python -m src.train_automl --time-budget 60     # FLAML AutoML + MLflow tracking
python -m src.evaluate                          # metrics on original test set
python -m src.make_modified_test                # build "academic stress" test set
python -m src.batch_inference                   # predict both sets + compare
python -m src.monitoring                        # PSI drift report

# Tests (requires pipeline artifacts to exist first)
pytest -q

# Serving
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
streamlit run app/dashboard.py                       # http://localhost:8501
uvicorn app.inference_api:app --port 8000            # http://localhost:8000/docs
```

## Architecture

**Single config source of truth:** `src/config.py` defines all paths, feature lists, random seed, MLflow URIs, and the `STRESS_SCENARIO` dict. Every script imports from it — change a value there and the whole pipeline stays consistent.

**Pipeline flow:**
```
raw CSV -> src/eda.py -> src/preprocessing.py -> src/train_automl.py
       -> src/evaluate.py -> src/make_modified_test.py -> src/batch_inference.py
       -> src/monitoring.py
```

**Model artifact:** `train_automl.py` builds a single sklearn `Pipeline(preprocessor → model)` — the `ColumnTransformer` (OHE for categoricals, passthrough for numerics) is baked in alongside the FLAML-chosen estimator. This means `app/` feeds raw student rows; no separate preprocessing step at serve time. The pipeline is saved both to MLflow registry and as `models/best_model.pkl` (so the app works without MLflow running).

**Monitoring:** `src/monitoring.py` computes PSI (Population Stability Index) between original and modified test prediction distributions. No heavy dependencies — the Evidently-based `src/monitoring_evidently.py` is an optional alternative for HTML reports.

**Modified test set:** `src/make_modified_test.py` applies `STRESS_SCENARIO` from config (halve study hours, spike stress, reduce sleep/attendance) to simulate academic stress. Used to demonstrate observable model degradation and drift detection.

**Deployment pair:** `app/inference_api.py` is a FastAPI REST endpoint; `app/dashboard.py` is a Streamlit app with two tabs — inference (single student) and monitoring (metric deltas + PSI drift table + prediction distribution chart).

## Key constraints

- **No leakage columns:** `Math_Score`, `Science_Score`, `Language_Score`, `History_Score`, `Grade` are never fed to the model — they are the direct components of `Average_Score` (the target).
- **MLflow tracking URI must be SQLite** (`sqlite:///mlflow.db`) to enable the Model Registry locally; a pure file store cannot register models.
- **`models/best_model.pkl` is the serving artifact** — the apps load this directly and must not depend on the MLflow server being live.
- `data/processed/` and `reports/` are generated; do not commit them. Run the pipeline to regenerate.

# ML_OPS Architecture — Student Performance Risk Pipeline

## Overview

End-to-end MLOps system that ingests a raw student dataset, applies staged data lineage, materializes versioned feature sets, runs AutoML-powered model selection, serves predictions via REST + dashboard, and monitors for data drift.

**Target:** Predict a continuous performance score per student and derive an at-risk flag (`score < RISK_THRESHOLD`).

---

## End-to-End Pipeline

```
data/raw/student_performance.csv          (DVC-tracked)
  │
  ▼ [src/lineage/run_lineage.py]
  ├─ data/staging/stg_students.parquet          ← Staging layer
  ├─ data/intermediate/int_student_features.parquet  ← Leakage stripped
  └─ data/marts/student_training.parquet        ← 80/20 split
     data/marts/student_serving.parquet
  │
  ▼ [src/feature_store.py]
  ├─ data/feature_store/v1/{train,test}.parquet  ← 14 original features
  └─ data/feature_store/v2/{train,test}.parquet  ← 14 engineered features
  │
  ▼ [src/experiment.py + src/mlflow_utils.py]
  └─ MLflow runs (feat version × hparams × CodeCarbon emissions)
  │
  ▼ [src/automl_benchmark.py]
  └─ models/best_model.pkl           ← Serving artifact (preprocessing baked in)
     models/model_metadata.json
  │
  ├─▶ [src/evaluate.py]              ← RMSE / MAE / R² / risk_accuracy / risk_f1
  ├─▶ [src/batch_inference.py]       ← Original + stress-test set comparison
  │
  ▼ [src/monitoring.py + src/monitoring_evidently.py]
  └─ reports/monitoring/             ← PSI per feature + Evidently HTML/Cloud
  │
  ▼ [app/inference_api.py + app/dashboard.py]
     POST /predict → predicted_score + at_risk bool
     Streamlit dashboard → interactive student scoring
```

---

## 1. Data Preparation — Staged Lineage

**Module:** `src/lineage/run_lineage.py`, `src/lineage/contracts.py`

A 4-layer warehouse pattern enforced with contract guards at every transition. Each guard raises `ValueError` on violation — bad data cannot silently propagate downstream.

```
Raw CSV
  └─▶ Staging       validate_staging()       — schema present, table non-empty
       └─▶ Intermediate  validate_intermediate()  — leakage/ID cols stripped, no nulls in numerics
            └─▶ Marts    validate_marts()          — leakage absent, target present, both partitions non-empty
```

**Leakage columns stripped at Intermediate stage:**
`Math_Score`, `Science_Score`, `Language_Score`, `History_Score`, `Grade`

These are downstream outcomes that would make the model trivially accurate but useless in production (a student's grade is not known before prediction time).

**Key guard call pattern** — every module that reads a downstream artifact calls its guard at the top:
```python
from src.lineage.contracts import validate_marts
validate_marts()   # halts if upstream artifact is missing or invalid
```

---

## 2. Feature Store — Versioned Feature Sets

**Module:** `src/feature_store.py`

`FeatureStore.materialize()` reads from the marts and writes two versioned parquet snapshots with accompanying metadata JSON.

| Version | Features | Notes |
|---------|----------|-------|
| **v1** | 14 (original) | `config.FEATURES` baseline |
| **v2** | 14 (2 added, 2 dropped) | Adds `study_sleep_ratio`, `effective_attendance`; drops `Scholarship_Status`, `Exam_Proximity` |

**Engineered features (v2):**
```python
study_sleep_ratio    = Study_Hours_Per_Day / Sleep_Hours_Per_Day
effective_attendance = Attendance_Rate × (1 − Stress_Level / 10.0)
```

**Directory layout:**
```
data/feature_store/
  v1/
    train.parquet
    test.parquet
    feature_defs.json   ← {version, features, n_features, n_train, n_test, train_hash}
  v2/
    train.parquet
    test.parquet
    feature_defs.json
```

**Consumer API:**
```python
X_train, X_test, y_train, y_test, meta = FeatureStore().get_feature_set("v2")
```

The `train_hash` field (MD5 of training feature matrix) enables reproducibility checks — any run can verify it was trained on the exact same data snapshot.

---

## 3. Experiment Tracking — MLflow + CodeCarbon

**Module:** `src/experiment.py`, `src/mlflow_utils.py`

Grid search over `{v1, v2} × hyperparameter combinations`. Every run is wrapped in `mlflow_utils.start_tracked_run` so carbon accounting is consistent across all experiments.

**MLflow configuration:**
- Backend: `sqlite:///mlflow.db` (enables local Model Registry without a tracking server process)
- UI: `http://localhost:5001`

**Per-run artifacts logged:**
- Feature version, feature list, feature hash
- All hyperparameters
- RMSE, MAE, R² metrics
- `dvc_data_hash` — links MLflow run to the exact DVC data version
- `kgCO2_eq` — carbon emissions via CodeCarbon

**Known gotcha:** CodeCarbon's logger resets to INFO on import. `logging.getLogger("codecarbon").setLevel(logging.ERROR)` must be called *inside* `track_emissions`, after the import line.

---

## 4. AutoML Model Selection — FLAML Benchmark

**Module:** `src/automl_benchmark.py`

Two tracks benchmarked head-to-head:

| Track | Description |
|-------|-------------|
| **Baseline** | sklearn Pipeline (StandardScaler + simple estimator) |
| **FLAML AutoML** | Budget-controlled search over algorithm families and hyperparameters |

FLAML budget controlled by `params.yaml → automl.flaml_budget` (default: 60 seconds).

**Output artifacts:**
```
models/
  best_model.pkl          ← sklearn-compatible pipeline, preprocessing baked in
  model_metadata.json     ← {best_algorithm, registered_model, thresholds, metrics}
reports/automl/
  benchmark.json          ← all candidate models with metrics + inference latency
```

**Critical design decision:** The winning pipeline includes the `ColumnTransformer` preprocessing step. Serving code calls `_model.predict(raw_rows)` directly — no preprocessing logic is duplicated outside the model.

The model is also registered in the MLflow Model Registry for lineage tracking, but the serving artifact is always `models/best_model.pkl` — the app never calls the MLflow server at prediction time.

---

## 5. Evaluation

**Module:** `src/evaluate.py`

Evaluated on two test sets:

| Test Set | Purpose |
|----------|---------|
| `data/processed/test.csv` | Original held-out split |
| `data/processed/test_modified.csv` | "Academic stress" synthetic set (built by `src/make_modified_test.py`) |

**Metrics computed:**

| Metric | Type | Role |
|--------|------|------|
| RMSE | Regression | Primary — penalizes large errors, same units as score |
| MAE | Regression | Average absolute miss in score points |
| R² | Regression | Share of variance explained |
| `risk_accuracy` | Classification | Accuracy of at-risk flag at threshold |
| `risk_f1` | Classification | F1 of at-risk flag (handles class imbalance) |

At-risk flag: `score < config.RISK_THRESHOLD` (threshold defined once in `src/config.py`).

`src/batch_inference.py` runs predictions on both test sets and writes `reports/predictions_comparison.csv` — this is the **distribution shift probe** before the formal monitoring stage.

---

## 6. Deployment — Inference Service + Dashboard

### FastAPI — `app/inference_api.py`

```
GET  /health         → {status, model, registered_model}
POST /predict        → {predicted_score, at_risk, risk_threshold}
POST /predict_batch  → [{predicted_score, at_risk}, ...]
```

- Model loaded once at startup via `joblib.load(config.MODEL_PATH)`
- Pydantic `Student` schema enforces types at the API boundary (14 raw input features)
- `predict_batch` endpoint used by monitoring callbacks for population-level drift checks
- Interactive Swagger UI at `http://localhost:8000/docs`

### Streamlit — `app/dashboard.py`

- Same `best_model.pkl` loaded at module level
- Sliders/dropdowns for each student feature → live predicted score + at-risk flag
- Renders EDA figures and monitoring reports from `reports/`
- Accessible at `http://localhost:8501`

### Input Schema (14 features)

| Feature | Type | Example |
|---------|------|---------|
| Age | int | 21 |
| Study_Hours_Per_Day | float | 4.0 |
| Sleep_Hours_Per_Day | float | 7.0 |
| Attendance_Rate | float | 80.0 |
| Stress_Level | float | 4.0 |
| Gender | str | "Female" |
| Major | str | "STEM" |
| Living_Area | str | "Urban" |
| Parent_Education | str | "Bachelor" |
| Scholarship_Status | str | "No" |
| Part_Time_Job | str | "No" |
| Extracurricular_Activities | str | "No" |
| Internet_Access | str | "Yes" |
| Exam_Proximity | str | "No" |

---

## 7. Monitoring — PSI + Evidently

### Track 1 — Population Stability Index (`src/monitoring.py`)

Dependency-free drift detection. Runs in CI without Evidently installed.

```python
psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float
```

**Bin edge strategy:** Quantile-based edges from the reference distribution (robust to outliers). Edge cases: near-constant features fall back to linear spacing.

**Interpretation:**

| PSI | Meaning |
|-----|---------|
| < 0.1 | Stable — no action needed |
| 0.1 – 0.2 | Slight shift — monitor |
| > 0.2 | Significant drift — investigate / retrain |

Compares training distribution (reference) against the live/modified test distribution (current), per feature.

### Track 2 — Evidently (`src/monitoring_evidently.py`)

- `DataDriftPreset` + `DataSummaryPreset` run as an Evidently `Report`
- Always writes local HTML to `reports/monitoring/`
- Cloud upload: if `EVIDENTLY_API_TOKEN` + `EVIDENTLY_PROJECT_ID` are in `.env`, report is pushed to Evidently Cloud; otherwise silently skipped
- Wrapped in `try/except ImportError` — never crashes the pipeline if Evidently is absent

**Credentials:** Copy `.env.example` → `.env` and fill in tokens. `.env` is gitignored — never committed.

---

## 8. Orchestration — DVC + Docker

### DVC Pipeline (`dvc.yaml`)

10-stage DAG. `dvc repro` skips unchanged stages using `dvc.lock` hashes.

```
eda
 └─ lineage
      └─ preprocessing
           └─ feature_store
                └─ experiment
                     └─ automl_benchmark
                          ├─ evaluate
                          ├─ make_modified_test
                          │    └─ batch_inference
                          └─ monitoring
                               └─ monitoring_evidently
```

**Key DVC commands:**
```bash
.venv/bin/dvc repro       # Reproduce full pipeline (skips unchanged stages)
.venv/bin/dvc status      # Show which stages are stale
.venv/bin/dvc params diff # Diff params.yaml vs last run
.venv/bin/dvc dag         # Print the pipeline DAG
```

**`params.yaml`** controls pipeline hyperparameters — `automl.flaml_budget` is the primary tuning knob.

**DVC-tracked files (commit these, not the generated artifacts):**
- `data/raw/student_performance.csv.dvc`
- `dvc.lock`
- `params.yaml`

### Docker (`docker-compose.yml`)

| Service | Command | Purpose |
|---------|---------|---------|
| `pipeline` | `docker compose run --rm pipeline` | Runs `dvc repro`; artifacts land on host via volume mount |
| `test` | `docker compose run --rm test` | Runs `pytest` against pipeline artifacts |
| `mlflow` | `docker compose up` | MLflow UI on `:5001` |
| `api` | `docker compose up` | FastAPI on `:8000` |
| `dashboard` | `docker compose up` | Streamlit on `:8501` |

```bash
bash scripts/verify_docker.sh   # all three steps (pipeline → test → serve) in sequence
bash serve_demo.sh               # local: mlflow + api + dashboard
```

---

## Single Source of Truth — `src/config.py`

Every path, feature list, threshold, and URI is defined once in `src/config.py` and imported everywhere. No values are hardcoded in individual scripts.

**Key constants:**
```
ROOT                  → project root (portable, relative to config.py location)
FEATURES              → 14-element list of input feature names
TARGET                → "GPA" (or equivalent performance score column)
RISK_THRESHOLD        → score below which a student is flagged at-risk
LEAKAGE_COLUMNS       → columns stripped at the Intermediate stage
MODEL_PATH            → models/best_model.pkl
MLFLOW_TRACKING_URI   → sqlite:///mlflow.db
FEATURE_STORE_DIR     → data/feature_store/
MONITORING_DIR        → reports/monitoring/
```

---

## Inference Flow (Runtime)

```
Client (Streamlit slider / API caller / batch job)
  │
  │  raw 14-feature student row (no preprocessing needed)
  ▼
POST /predict  (app/inference_api.py)
  │
  │  Pydantic validation → pd.DataFrame
  ▼
_model.predict(row)   ← models/best_model.pkl
  │                      (ColumnTransformer + FLAML estimator, loaded once at startup)
  ▼
predicted_score (float)
  │
  ├─ at_risk = score < RISK_THRESHOLD  (bool)
  └─ return {predicted_score, at_risk, risk_threshold}
```

---

## Tool Stack Summary

| Pillar | Tool | Role |
|--------|------|------|
| Data Lineage | DVC + custom contracts | Staged transforms, artifact versioning, contract guards |
| Feature Store | pandas + parquet | Versioned (v1/v2) feature snapshots with metadata |
| Experiment Tracking | MLflow (SQLite) + CodeCarbon | Run tracking, Model Registry, emissions |
| AutoML Optimization | FLAML | Budget-controlled algorithm + hyperparameter search |
| Serving | FastAPI + Streamlit | REST inference + interactive demo |
| Drift Monitoring | PSI (custom) + Evidently | Population stability + full drift reports |
| Orchestration | DVC + Docker Compose | Reproducible pipeline DAG + containerized execution |
| Config | `src/config.py` | Single source of truth for all constants |

# Student Performance Risk Monitoring System

[![CI](https://github.com/campbelltaylor32/ML_OPS/actions/workflows/ci.yml/badge.svg)](https://github.com/campbelltaylor32/ML_OPS/actions/workflows/ci.yml)

An end-to-end **Machine Learning Operations (MLOps)** project that predicts a
student's **Performance Score** from demographic, background, and behavioural
features, and flags students who may be **At Risk** so academic advisors can
intervene early.

The system demonstrates the **full MLOps lifecycle** across four pillars:
**Data Lineage → Feature Store + Experiment Tracking → AutoML Optimization → Drift Monitoring**.

---

## Live Services & URLs

Run `bash serve_demo.sh` to start all apps and print every URL. Or open
`app/serve_index.html` in your browser for a one-click landing page.

| Service | URL | Notes |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | Predict + Monitoring tabs |
| FastAPI Swagger UI | http://localhost:8000/docs | Interactive inference API |
| FastAPI health probe | http://localhost:8000/health | Liveness check |
| MLflow UI | http://localhost:5001 | Experiments + model registry |
| Evidently drift report | `reports/monitoring/evidently_report.html` | Static HTML |
| AutoML benchmark chart | `reports/automl/accuracy_vs_latency.png` | Static PNG |
| Lineage DAG | `reports/lineage/lineage_dag.png` | Static PNG |
| All EDA figures | `reports/figures/` | Static PNGs |

---

## End-to-End Run & Demo Guide

Follow these steps in order for the recorded demo. Everything runs locally — no cloud account needed.

### Step 1 — Setup (once)

```bash
uv sync                       # creates .venv and installs all deps from uv.lock
cp .env.example .env          # fill in EVIDENTLY_API_TOKEN if you have one (optional)
```

### Step 2 — Run the full pipeline

```bash
bash run_all.sh
```

This runs all 11 steps in order: EDA → lineage → preprocessing → feature store →
experiment grid → AutoML benchmark → evaluate → build modified test set → batch
inference → PSI monitoring → Evidently drift report.

Expected output: `Done.` with the three serving commands printed at the end.
Artifacts generated: `models/best_model.pkl`, `reports/`, `data/marts/`,
`data/feature_store/`, `data/processed/`.

### Step 3 — Verify everything passes

```bash
uv run python -m pytest -q
```

All 17 guard tests (G1–G4) should pass. If any fail, re-run `bash run_all.sh` first.

### Step 4 — Launch all services

```bash
bash serve_demo.sh
```

Starts MLflow UI, FastAPI, and Streamlit together and prints every URL. Press
**Ctrl-C** to stop all three at once. Keep this terminal open for the recording.

### Step 5 — Demo walkthrough (record in this order)

| # | What to show | Where |
|---|---|---|
| 1 | **EDA** — target distribution, feature correlations | `reports/figures/` (open PNGs) |
| 2 | **Pipeline + AutoML results** — experiment runs, registered model | http://localhost:5001 → Experiments → student_performance_model |
| 3 | **Live inference (original data)** — predict a student, see score + risk flag | http://localhost:8501 → Predict tab |
| 4 | **Monitoring — original test set** — RMSE, R², mean predicted score | http://localhost:8501 → Monitoring tab (top metrics) |
| 5 | **Changed test set results** — same tab shows modified set metrics side-by-side, PSI drift table | http://localhost:8501 → Monitoring tab (scroll down) |
| 6 | **Evidently drift report** — DataDrift + DataSummary HTML | `reports/monitoring/evidently_report.html` (open in browser) |
| 7 | **REST API** — hit `/predict` live via Swagger | http://localhost:8000/docs → POST /predict → Try it out |
| 8 | **All URLs at a glance** | `app/serve_index.html` (open in browser) |

> **Tip:** open `app/serve_index.html` first and use it as your navigation hub
> throughout the recording — every link is one click from there.

---

## 1. Project title
**Student Performance Risk Monitoring System — Predicting and Monitoring Academic Performance with MLOps**

## 2. Business problem statement
> Academic advisors cannot personally track the workload, sleep, stress, and
> attendance of every student, so at-risk students are often identified only
> after grades have already slipped. This project delivers an automated risk
> monitoring system that predicts each student's performance score from
> behaviour and background, flags those likely to fall below a passing score,
> and continuously monitors whether incoming student data has drifted from what
> the model was trained on. Advisors get an early, data-driven shortlist of
> students who need support — and the institution gets confidence that the model
> stays reliable as the student population changes.

## 3. Dataset & target
- **1,000 student records**, no missing values (`data/raw/student_performance.csv`).
- **Primary target (regression):** `Average_Score` — the Performance Score, 0–100.
- **Optional risk flag (classification view):** `At Risk = predicted score < 60`.
- **Features used (14 base + 2 derived in v2):**
  - *Numeric:* Age, Study_Hours_Per_Day, Sleep_Hours_Per_Day, Attendance_Rate, Stress_Level
  - *Categorical:* Gender, Major, Living_Area, Parent_Education, Scholarship_Status, Part_Time_Job, Extracurricular_Activities, Internet_Access, Exam_Proximity
  - *Derived (v2):* `study_sleep_ratio`, `effective_attendance`
- **Deliberately excluded (target leakage):** `Math_Score`, `Science_Score`,
  `Language_Score`, `History_Score`, `Grade`. `Average_Score` is literally the
  mean of the four subject scores, so feeding them in would let the model trivially
  re-average them (R² ≈ 1.0) and teach it nothing useful. `Name` is dropped as a
  non-predictive identifier.

---

## 4. MLOps Pillars

### Pillar 1 — Data Lineage (`src/lineage/`)
dbt-style staged transforms; each stage writes parquet and is validated by `contracts.py`
before the next stage runs.

```
data/raw/student_performance.csv
  └─[stg_students.py]──▶ data/staging/stg_students.parquet          (hygiene only)
       └─[int_features.py]──▶ data/intermediate/int_student_features.parquet  (drop leakage, add derived)
            └─[mart_training.py]──▶ data/marts/student_{training,serving}.parquet  (fixed-seed split)
```

Lineage DAG diagram: `reports/lineage/lineage_dag.{md,png}`

**Checking Guard G1:** `contracts.validate_marts()` must pass before features are built.

### Pillar 2 — Feature Store + Experiment Tracking (`src/feature_store.py`, `src/experiment.py`)
- **`FeatureStore`** materialises two versioned feature sets from the mart:
  - `v1` — 14 original features
  - `v2` — 14 features + `study_sleep_ratio`, `effective_attendance` (2 derived removed 2 weak categoricals)
- **`src/experiment.py`** runs a 6-run grid (2 versions × 3 hyperparameter configs) with `codecarbon`
  emissions tracking baked in. Every MLflow run records `co2_kg_emissions` as a metric.
- **`src/mlflow_utils.py`** provides `start_tracked_run`, `log_feature_set`, and `track_emissions`
  context manager — reusable across all training scripts.

**Checking Guard G2:** feature store `feature_defs.json` must exist for both versions before AutoML.

### Pillar 3 — AutoML Optimization (`src/automl_benchmark.py`)
Benchmarks four regimes on identical splits, all logged to MLflow with latency:

| Regime | Description |
|---|---|
| `baseline` | Default LightGBM, no tuning |
| `flaml` | FLAML AutoML (lgbm / xgboost / rf / extra_tree) |
| `h2o` | H2O AutoML (JVM-guarded; skips cleanly if unavailable) |
| `top5` | Top-5 features by importance, LightGBM retrained |

Results + accuracy-vs-latency scatter: `reports/automl/benchmark.json` + `accuracy_vs_latency.png`

**Checking Guard G3:** `models/best_model.pkl` + `reports/automl/benchmark.json` must exist before monitoring.

### Pillar 4 — Drift Monitoring
Two monitoring tracks run in parallel:

| Module | What it does |
|---|---|
| `src/monitoring.py` | PSI drift, dependency-free, always works offline |
| `src/monitoring_evidently.py` | Evidently `DataDriftPreset` + `DataSummaryPreset`; HTML always saved; Cloud upload when `EVIDENTLY_API_TOKEN` env var is set |

**Checking Guard G4:** reference (`test.csv`) and current (`test_modified.csv`) must exist.

---

## 5. Folder structure
```
student-performance-mlops/
├── data/
│   ├── raw/student_performance.csv            # input dataset
│   ├── staging/stg_students.parquet           # Lineage stage 1 (generated)
│   ├── intermediate/int_student_features.parquet  # Lineage stage 2 (generated)
│   ├── marts/student_{training,serving}.parquet   # Lineage stage 3 (generated)
│   ├── feature_store/{v1,v2}/                 # Versioned feature sets (generated)
│   └── processed/                             # CSV compat split (generated)
├── src/
│   ├── config.py                              # single source of truth (all paths, columns, seed, env vars)
│   ├── lineage/
│   │   ├── stg_students.py                    # Raw → Staging
│   │   ├── int_features.py                    # Staging → Intermediate
│   │   ├── mart_training.py                   # Intermediate → Marts
│   │   ├── contracts.py                       # Guard functions G0–G3 (raise on violation)
│   │   ├── lineage_diagram.py                 # Mermaid + PNG DAG
│   │   └── run_lineage.py                     # Orchestrator
│   ├── mlflow_utils.py                        # start_tracked_run, log_feature_set, track_emissions
│   ├── feature_store.py                       # FeatureStore v1/v2 materialisation
│   ├── experiment.py                          # Feature grid × hyperparams + CodeCarbon
│   ├── automl_benchmark.py                    # Baseline vs FLAML vs H2O + latency + top-5
│   ├── eda.py                                 # EDA charts
│   ├── preprocessing.py                       # CSV-compat train/test split
│   ├── train_automl.py                        # Standalone FLAML AutoML + MLflow
│   ├── evaluate.py                            # Metrics on original test set
│   ├── make_modified_test.py                  # "Academic stress" modified test set
│   ├── batch_inference.py                     # Predict both sets + compare
│   ├── monitoring.py                          # PSI drift monitor (dependency-free)
│   └── monitoring_evidently.py               # Evidently report (DataDrift + DataSummary + Cloud)
├── app/
│   ├── inference_api.py                       # FastAPI REST endpoint
│   └── dashboard.py                           # Streamlit inference + monitoring dashboard
├── tests/
│   ├── test_pipeline.py                       # Original smoke tests
│   └── test_mlops_pillars.py                  # G1–G4 guard tests (17 tests total)
├── reports/
│   ├── figures/                               # EDA charts (generated)
│   ├── lineage/lineage_dag.{md,png}           # Lineage DAG (generated)
│   ├── experiments/feature_grid.json          # Feature grid results (generated)
│   ├── automl/benchmark.json                  # AutoML benchmark (generated)
│   └── monitoring/                            # drift_report.json, evidently_report.html (generated)
├── models/best_model.pkl                      # Serving artifact (generated)
├── mlflow.db / mlruns/                        # MLflow tracking store (generated)
├── run_all.sh                                 # 11-step end-to-end pipeline
├── requirements.txt
├── lessons_learned.md
└── README.md
```

---

## 6. Local setup

```bash
uv sync              # creates .venv at project root; installs all deps from uv.lock
uv sync --extra h2o  # also install H2O (requires JVM 8+)

# Evidently Cloud credentials (optional — local HTML report always works without them)
cp .env.example .env          # then open .env and fill in your token + project ID
```

`.env` is gitignored — never commit it. `src/config.py` loads it automatically via
`python-dotenv`, so no `export` commands are needed.

### Dependency files
| File | Purpose |
|---|---|
| `pyproject.toml` | Direct dependencies with pinned versions — edit when upgrading a package |
| `uv.lock` | Full dependency lock (all transitive deps) — guarantees exact reproduction |

To upgrade a package: bump its version in `pyproject.toml`, then regenerate the lock:
```bash
uv lock
uv sync
bash run_all.sh
uv run python -m pytest -q
```

## 7. Run the whole pipeline (one command)
```bash
bash run_all.sh
```

Or run each pillar individually:
```bash
# Pillar 1 — Data Lineage
uv run python -m src.lineage.run_lineage

# Pillar 2 — Feature Store + Experiments
uv run python -m src.feature_store
uv run python -m src.experiment

# Pillar 3 — AutoML Benchmark
uv run python -m src.automl_benchmark --flaml-budget 60

# Pillar 4 — Monitoring
uv run python -m src.monitoring
uv run python -m src.monitoring_evidently

# Guard tests
uv run python -m pytest -q
```

## 8. Deploy & monitor
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # http://localhost:5001
uv run streamlit run app/dashboard.py                                   # http://localhost:8501
uv run uvicorn app.inference_api:app --port 8000                        # http://localhost:8000/docs
```

## 8b. Docker deployment (recommended for reproducible demos)

Requires Docker ≥ 24 + Docker Compose v2.

```bash
# First run — build pipeline artifacts inside the container (bind-mount writes to host)
docker compose run --rm pipeline

# Start all three serving services
docker compose up

# Individual service
docker compose up api          # FastAPI only (http://localhost:8000/docs)
docker compose up mlflow       # MLflow UI only (http://localhost:5001)
docker compose up dashboard    # Streamlit only (http://localhost:8501)
```

The `pipeline` service (profile: `pipeline`) is excluded from `docker compose up` so it never auto-starts. Run it once to generate `models/best_model.pkl` and all report artifacts before launching the serving stack.

Health endpoint verified by the compose healthcheck:
```
GET http://localhost:8000/health → {"status":"ok","model":"xgboost",...}
```

## 9. MLflow tracking setup
- Tracking store: **local SQLite** (`sqlite:///mlflow.db`), configured in `src/config.py`.
- Every training run (experiment grid + AutoML benchmark) logs:
  - Parameters: feature version, hyperparameters, algorithm
  - Metrics: RMSE, MAE, R², training time (s), inference latency (ms)
  - **Carbon emissions:** `co2_kg_emissions` via CodeCarbon (tagged `carbon_tracked=true`)
- Best FLAML pipeline registered as `student_performance_model` (versioned).
- `models/best_model.pkl` is the serving artifact — apps load this directly.

## 10. Evidently Cloud setup (optional)

1. Copy the template: `cp .env.example .env`
2. Edit `.env` and set:
   - `EVIDENTLY_API_TOKEN` — from **app.evidently.cloud → Settings → API Keys**
   - `EVIDENTLY_PROJECT_ID` — project UUID (optional; first workspace project used if blank)
3. Run: `python -m src.monitoring_evidently`

`src/config.py` loads `.env` automatically via `python-dotenv` — no `export` needed.
Without a token, the script writes `reports/monitoring/evidently_report.html` locally and skips the upload.

---

## Results (from a 60-second FLAML AutoML run)

### AutoML Benchmark
| Regime | RMSE | R² | Latency (ms) |
|---|---|---|---|
| Baseline LightGBM | ~4.70 | ~0.80 | ~1 ms |
| FLAML AutoML | ~4.08 | ~0.85 | ~2 ms |
| Top-5 Features | ~5.18 | ~0.76 | ~0.5 ms |
| H2O AutoML | skipped (JVM not available) | — | — |

### Drift Detection (original vs. "academic stress" modified set)
| Metric | Original | Modified | Change |
|---|---|---|---|
| **RMSE** | ~3.95 | ~19.6 | ▲ degraded |
| **R²** | ~0.86 | ~-2.4 | ▼ degraded |
| **Mean predicted score** | ~71.2 | ~53.2 | ▼ ~18 pts |
| **% flagged At Risk** | ~14% | ~70% | ▲ +56 pts |
| **Prediction drift (PSI)** | — | ~3.97 | **major drift** |

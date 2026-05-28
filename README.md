# Student Performance Risk Monitoring System

An end-to-end **Machine Learning Operations (MLOps)** project that predicts a
student's **Performance Score** from demographic, background, and behavioural
features, and flags students who may be **At Risk** so academic advisors can
intervene early.

The system demonstrates the **full MLOps lifecycle** across four pillars:
**Data Lineage → Feature Store + Experiment Tracking → AutoML Optimization → Drift Monitoring**.

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
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Exact reproduction (recommended) — installs every transitive dep at a pinned version
pip install -r requirements.lock

# Direct deps only — resolves transitive deps fresh (may pull newer sub-deps)
# pip install -r requirements.txt

# Optional: install H2O for the H2O AutoML benchmark (requires Java 8+)
# pip install h2o

# Evidently Cloud credentials (optional — local HTML report always works without them)
cp .env.example .env          # then open .env and fill in your token + project ID
```

`.env` is gitignored — never commit it. `src/config.py` loads it automatically via
`python-dotenv`, so no `export` commands are needed.

### Dependency files
| File | Purpose |
|---|---|
| `requirements.txt` | Pinned direct dependencies — edit this when upgrading a package |
| `requirements.lock` | Full `pip freeze` snapshot — guarantees byte-for-byte reproducible installs |

To upgrade a package: bump its version in `requirements.txt`, install, run the full test suite, then regenerate the lock file:
```bash
pip install -r requirements.txt
bash run_all.sh
pip freeze > requirements.lock
```

## 7. Run the whole pipeline (one command)
```bash
bash run_all.sh
```

Or run each pillar individually:
```bash
# Pillar 1 — Data Lineage
python -m src.lineage.run_lineage

# Pillar 2 — Feature Store + Experiments
python -m src.feature_store
python -m src.experiment

# Pillar 3 — AutoML Benchmark
python -m src.automl_benchmark --flaml-budget 60

# Pillar 4 — Monitoring
python -m src.monitoring
python -m src.monitoring_evidently

# Guard tests
python -m pytest -q
```

## 8. Deploy & monitor
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
streamlit run app/dashboard.py                       # http://localhost:8501
uvicorn app.inference_api:app --port 8000            # http://localhost:8000/docs
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

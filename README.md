# Student Performance Risk Monitoring System

An end-to-end **Machine Learning Operations (MLOps)** project that predicts a
student's **Performance Score** from demographic, background, and behavioural
features, and flags students who may be **At Risk** so academic advisors can
intervene early. It demonstrates the full MLOps lifecycle on easy local tools:
EDA → reproducible split → AutoML training with experiment tracking → model
registry → deployment → monitoring → drift validation.

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
- **Features used (14):**
  - *Numeric:* Age, Study_Hours_Per_Day, Sleep_Hours_Per_Day, Attendance_Rate, Stress_Level
  - *Categorical:* Gender, Major, Living_Area, Parent_Education, Scholarship_Status, Part_Time_Job, Extracurricular_Activities, Internet_Access, Exam_Proximity
- **Deliberately excluded (target leakage):** `Math_Score`, `Science_Score`,
  `Language_Score`, `History_Score`, `Grade`. `Average_Score` is literally the
  mean of the four subject scores, so feeding them in would let the model trivially
  re-average them (R² ≈ 1.0) and teach it nothing useful. `Name` is dropped as a
  non-predictive identifier. Excluding these forces the model to learn from
  factors advisors can actually influence (study time, sleep, stress, attendance).

---

## 4. Folder structure
```
student-performance-mlops/
├── data/
│   ├── raw/student_performance.csv        # input dataset
│   └── processed/                         # train.csv, test.csv, test_modified.csv (generated)
├── src/
│   ├── config.py                          # single source of truth (paths, columns, seed, scenario)
│   ├── eda.py                             # [2] EDA charts
│   ├── preprocessing.py                   # [3] clean + train/test split
│   ├── train_automl.py                    # [4][5] FLAML AutoML + MLflow tracking + registry
│   ├── evaluate.py                        # [6] metrics on original test set
│   ├── make_modified_test.py              # [9] build the "academic stress" test set
│   ├── batch_inference.py                 # [8][10][16] predict + compare original vs modified
│   ├── monitoring.py                      # [7][12] PSI drift monitor (dependency-free)
│   └── monitoring_evidently.py            # [7] optional Evidently HTML report
├── app/
│   ├── inference_api.py                   # [6] FastAPI deployment (REST)
│   └── dashboard.py                       # [7][12] Streamlit inference + monitoring dashboard
├── tests/test_pipeline.py                 # smoke tests (no leakage, range, stress lowers score)
├── reports/                               # figures/, monitoring/, comparison json+csv (generated)
├── models/                                # best_model.pkl + model_metadata.json (generated)
├── mlflow.db / mlruns/                    # MLflow tracking store + artifacts (generated)
├── run_all.sh                             # one-command end-to-end pipeline
├── requirements.txt
└── README.md
```

---

## 5. Local setup
```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. confirm the dataset is in place
ls data/raw/student_performance.csv
```

## 6. Run the whole pipeline (one command)
```bash
bash run_all.sh
```
Or run each step individually (all are `python -m src.<name>`):
```bash
python -m src.eda                       # [2] charts -> reports/figures/
python -m src.preprocessing             # [3] -> data/processed/train.csv, test.csv
python -m src.train_automl --time-budget 60   # [4][5] AutoML + MLflow + registry
python -m src.evaluate                  # [6] metrics on original test set
python -m src.make_modified_test        # [9] modified "stress" test set
python -m src.batch_inference           # [8][10] predict on both + compare
python -m src.monitoring                # [7][12] drift report
pytest -q                               # smoke tests
```

## 7. Deploy & monitor
```bash
# Experiment tracking + model registry UI
mlflow ui --backend-store-uri sqlite:///mlflow.db          # http://localhost:5000

# Interactive inference + monitoring dashboard (record THIS for the video demo)
streamlit run app/dashboard.py                             # http://localhost:8501

# REST API for programmatic inference
uvicorn app.inference_api:app --port 8000                  # http://localhost:8000/docs
```

## 8. MLflow tracking setup
- Tracking store: **local SQLite** (`sqlite:///mlflow.db`) — set in `src/config.py`,
  which enables the **Model Registry** locally (a pure file store cannot register models).
- Each training run logs: parameters (time budget, seed, feature count), the
  AutoML-chosen algorithm + best hyperparameters, test metrics (RMSE/MAE/R²), and
  the full **preprocessing+model pipeline** as an artifact.
- The best pipeline is registered as **`student_performance_model`** (versioned).
- A plain `models/best_model.pkl` is also saved so the app/dashboard serve the
  model without needing the MLflow server running during the demo.

---

## Results (from a 60-second AutoML run)
AutoML (FLAML) compared LightGBM, XGBoost, Random Forest, and Extra Trees and
selected **XGBoost**.

| Metric | Original test set | Modified ("stress") test set | Change |
|---|---|---|---|
| **RMSE** | ~3.95 | ~19.6 | ▲ degraded |
| **MAE** | ~2.85 | ~18.2 | ▲ degraded |
| **R²** | ~0.86 | ~-2.4 | ▼ degraded |
| **Mean predicted score** | ~71.2 | ~53.2 | ▼ ~18 points |
| **% flagged At Risk** | ~14% | ~70% | ▲ +56 pts |
| **Prediction drift (PSI)** | — | ~3.97 (**major drift**) | detected |

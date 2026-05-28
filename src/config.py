"""
config.py
=========
Single source of truth for the whole project.

MLOps purpose: REPRODUCIBILITY. Every script (EDA, preprocessing, training,
evaluation, inference, monitoring) imports paths, column groups, the random
seed, and modeling decisions from this one file. Nothing is hard-coded twice,
so a teammate can change one value here and the entire pipeline stays
consistent.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the project root so the repo is portable)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "student_performance.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
TEST_MODIFIED_PATH = PROCESSED_DIR / "test_modified.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"          # plain pickle for the app
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DIR = REPORTS_DIR / "monitoring"
PREDICTIONS_PATH = REPORTS_DIR / "predictions_comparison.csv"
METRICS_PATH = REPORTS_DIR / "metrics_comparison.json"

# MLflow tracking + registry (sqlite backend so the Model Registry works locally)
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "student_performance_risk"
REGISTERED_MODEL_NAME = "student_performance_model"

# --------------------------------------------------------------------------- #
# Modeling decisions
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20

# Primary regression target (the "Performance Score", 0-100)
TARGET = "Average_Score"

# Behavioral + demographic + background features used to PREDICT the target.
NUMERIC_FEATURES = [
    "Age",
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Attendance_Rate",
    "Stress_Level",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Major",
    "Living_Area",
    "Parent_Education",
    "Scholarship_Status",
    "Part_Time_Job",
    "Extracurricular_Activities",
    "Internet_Access",
    "Exam_Proximity",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns we DROP and never feed to the model.
#   - Average_Score is literally the mean of the four subject scores, so using
#     the subject scores (or Grade) as inputs is TARGET LEAKAGE: the model would
#     just re-average them and score a meaningless R2 ~ 1.0.
#   - Name is a unique identifier with no predictive value (and is PII).
# Excluding them forces the model to learn from behaviour/background, which is
# exactly what academic advisors can actually influence.
LEAKAGE_COLUMNS = [
    "Math_Score",
    "Science_Score",
    "Language_Score",
    "History_Score",
    "Grade",
]
ID_COLUMNS = ["Name"]

# Optional risk flag derived from the predicted score (no second model needed).
# A student is "At Risk" if their predicted Performance Score is below this.
RISK_THRESHOLD = 60.0

# --------------------------------------------------------------------------- #
# "Academic stress" scenario for the modified test set (assignment step 9).
# Each entry: column -> (operation, value). >=2 features are changed.
# --------------------------------------------------------------------------- #
STRESS_SCENARIO = {
    "Study_Hours_Per_Day": ("multiply", 0.5),   # halve study time
    "Stress_Level":        ("add", 3.0),         # spike stress
    "Sleep_Hours_Per_Day": ("add", -1.5),        # lose sleep
    "Attendance_Rate":     ("add", -15.0),       # skip classes
}
# Clip ranges so modified values stay physically plausible.
CLIP_RANGES = {
    "Study_Hours_Per_Day": (0.0, 24.0),
    "Stress_Level":        (1.0, 10.0),
    "Sleep_Hours_Per_Day": (0.0, 14.0),
    "Attendance_Rate":     (0.0, 100.0),
}


def ensure_dirs() -> None:
    """Create output directories if they do not exist (safe to call anytime)."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, MONITORING_DIR):
        d.mkdir(parents=True, exist_ok=True)

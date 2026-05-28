"""
evaluate.py  --  Step 6: evaluate the deployed model on the original test set
=============================================================================
MLOps purpose: MODEL VALIDATION against the agreed metric (assignment step 3).
Defines and reports the evaluation metrics, plus the derived "At Risk" flag
quality, and saves a residual plot for the slides.

Metric definition (assignment step 3):
  Primary  -> RMSE  (penalises large errors; same units as the score)
  Support  -> MAE   (average absolute error in score points)
  Support  -> R^2   (share of score variance explained by the model)

Run:  python -m src.evaluate
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src import config


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate():
    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    X, y = test_df[config.FEATURES], test_df[config.TARGET]

    preds = model.predict(X)
    metrics = regression_metrics(y, preds)

    # Optional classification view: At Risk = score < threshold
    true_risk = (y < config.RISK_THRESHOLD).astype(int)
    pred_risk = (preds < config.RISK_THRESHOLD).astype(int)
    metrics["risk_accuracy"] = float(accuracy_score(true_risk, pred_risk))
    metrics["risk_f1"] = float(f1_score(true_risk, pred_risk, zero_division=0))

    print("=== Evaluation on ORIGINAL test set ===")
    print(json.dumps(metrics, indent=2))

    # Predicted vs actual scatter for the slides
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, preds, s=14, alpha=0.4, color="#4C72B0")
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    ax.plot(lims, lims, "--", color="#C44E52", label="perfect prediction")
    ax.set(
        title="Predicted vs Actual Performance Score (test set)",
        xlabel="Actual",
        ylabel="Predicted",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_pred_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    (config.REPORTS_DIR / "eval_original.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()

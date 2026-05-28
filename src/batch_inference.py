"""
batch_inference.py  --  Steps 8, 10 & 16: run the model on both test sets
==========================================================================
MLOps purpose: VALIDATE with the deployed model and COMPARE outputs before vs
after the data change. Loads the same pickle the app serves, predicts on the
original and the modified test sets, computes RMSE/MAE/R2 for each, and saves a
side-by-side comparison (table + chart) for the monitoring dashboard and slides.

Run:  python -m src.batch_inference
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.evaluate import regression_metrics


def _predict(model, path):
    df = pd.read_csv(path)
    preds = model.predict(df[config.FEATURES])
    metrics = regression_metrics(df[config.TARGET], preds)
    return df, preds, metrics


def run_comparison():
    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)

    orig_df, orig_pred, orig_m = _predict(model, config.TEST_PATH)
    mod_df, mod_pred, mod_m = _predict(model, config.TEST_MODIFIED_PATH)

    comparison = {
        "original": {
            **orig_m,
            "mean_prediction": float(orig_pred.mean()),
            "at_risk_pct": float((orig_pred < config.RISK_THRESHOLD).mean() * 100),
        },
        "modified": {
            **mod_m,
            "mean_prediction": float(mod_pred.mean()),
            "at_risk_pct": float((mod_pred < config.RISK_THRESHOLD).mean() * 100),
        },
    }
    comparison["delta"] = {
        "mean_prediction_drop": round(
            comparison["original"]["mean_prediction"] - comparison["modified"]["mean_prediction"], 2
        ),
        "at_risk_pct_increase": round(
            comparison["modified"]["at_risk_pct"] - comparison["original"]["at_risk_pct"], 2
        ),
        "rmse_increase": round(comparison["modified"]["RMSE"] - comparison["original"]["RMSE"], 2),
    }

    # Per-student prediction table
    table = pd.DataFrame(
        {
            "actual": orig_df[config.TARGET],
            "pred_original": orig_pred,
            "pred_modified": mod_pred,
            "pred_change": mod_pred - orig_pred,
        }
    )
    table.to_csv(config.PREDICTIONS_PATH, index=False)
    config.METRICS_PATH.write_text(json.dumps(comparison, indent=2))

    # Overlaid prediction distributions -> the money chart for the demo
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(orig_pred, bins=25, alpha=0.6, label="Original", color="#4C72B0")
    ax.hist(mod_pred, bins=25, alpha=0.6, label="Modified (stress)", color="#C44E52")
    ax.axvline(
        config.RISK_THRESHOLD,
        color="black",
        linestyle="--",
        label=f"At-Risk (<{config.RISK_THRESHOLD:.0f})",
    )
    ax.set(
        title="Predicted Performance Score: Original vs Modified test set",
        xlabel="Predicted score",
        ylabel="Number of students",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "08_prediction_shift.png", bbox_inches="tight")
    plt.close(fig)

    print("=== Original vs Modified comparison ===")
    print(json.dumps(comparison, indent=2))
    print(f"Saved table   -> {config.PREDICTIONS_PATH.relative_to(config.ROOT)}")
    print(f"Saved metrics -> {config.METRICS_PATH.relative_to(config.ROOT)}")
    return comparison


if __name__ == "__main__":
    run_comparison()

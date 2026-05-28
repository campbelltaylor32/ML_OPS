"""
monitoring.py  --  Steps 7 & 12: data + prediction drift monitoring
====================================================================
MLOps purpose: MONITORING. Detects whether incoming data has drifted from the
reference (training/original) distribution and whether model predictions have
shifted. Uses the Population Stability Index (PSI), a standard, dependency-free
drift metric, so the demo never breaks on a library version mismatch.

PSI interpretation (industry convention):
    PSI < 0.10  -> no significant drift
    0.10-0.25   -> moderate drift, investigate
    PSI > 0.25  -> major drift, model may be unreliable

This module powers both the Streamlit dashboard (app/dashboard.py) and a saved
HTML report. For an Evidently-based report see src/monitoring_evidently.py.

Run:  python -m src.monitoring
"""

import json

import joblib
import numpy as np
import pandas as pd

from src import config


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    # Bin edges from reference quantiles (robust to outliers)
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if len(edges) < 3:  # near-constant feature
        edges = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def label(score: float) -> str:
    if score < 0.10:
        return "no drift"
    if score < 0.25:
        return "moderate drift"
    return "major drift"


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, model=None) -> dict:
    """Feature-level PSI + optional prediction PSI between two datasets."""
    report = {"features": {}, "summary": {}}

    for col in config.NUMERIC_FEATURES:
        score = psi(reference_df[col].values, current_df[col].values)
        report["features"][col] = {
            "psi": round(score, 4),
            "status": label(score),
            "ref_mean": round(float(reference_df[col].mean()), 2),
            "cur_mean": round(float(current_df[col].mean()), 2),
        }

    # Categorical drift = change in category proportions (also via PSI on codes)
    for col in config.CATEGORICAL_FEATURES:
        cats = sorted(set(reference_df[col]) | set(current_df[col]))
        codes = {c: i for i, c in enumerate(cats)}
        score = psi(
            reference_df[col].map(codes).values,
            current_df[col].map(codes).values,
            bins=min(len(cats), 10),
        )
        report["features"][col] = {"psi": round(score, 4), "status": label(score)}

    drifted = [c for c, v in report["features"].items() if v["psi"] >= 0.10]
    report["summary"]["n_features"] = len(report["features"])
    report["summary"]["n_drifted"] = len(drifted)
    report["summary"]["drifted_features"] = drifted
    report["summary"]["dataset_drift"] = len(drifted) > 0

    # Prediction drift
    if model is not None:
        ref_pred = model.predict(reference_df[config.FEATURES])
        cur_pred = model.predict(current_df[config.FEATURES])
        report["prediction"] = {
            "psi": round(psi(ref_pred, cur_pred), 4),
            "status": label(psi(ref_pred, cur_pred)),
            "ref_mean_pred": round(float(ref_pred.mean()), 2),
            "cur_mean_pred": round(float(cur_pred.mean()), 2),
            "ref_at_risk_pct": round(float((ref_pred < config.RISK_THRESHOLD).mean() * 100), 1),
            "cur_at_risk_pct": round(float((cur_pred < config.RISK_THRESHOLD).mean() * 100), 1),
        }
    return report


def run_monitoring():
    config.ensure_dirs()
    model = joblib.load(config.MODEL_PATH)
    reference = pd.read_csv(config.TEST_PATH)  # original = reference
    current = pd.read_csv(config.TEST_MODIFIED_PATH)  # modified = production

    report = compute_drift(reference, current, model)
    out = config.MONITORING_DIR / "drift_report.json"
    out.write_text(json.dumps(report, indent=2))

    print("=== Drift report (original reference vs modified current) ===")
    print(json.dumps(report["summary"], indent=2))
    if "prediction" in report:
        print("Prediction drift:", json.dumps(report["prediction"], indent=2))
    print(f"Saved -> {out.relative_to(config.ROOT)}")
    return report


if __name__ == "__main__":
    run_monitoring()

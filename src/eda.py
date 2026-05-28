"""
eda.py  --  Step 2 of the pipeline: Exploratory Data Analysis
=============================================================
MLOps purpose: DATA UNDERSTANDING. Before training anything we profile the data
and save charts to reports/figures/. These figures go straight into the EDA
slides of the presentation and help justify the leakage decision in config.py.

Run:  python -m src.eda
"""
import json
import matplotlib
matplotlib.use("Agg")  # headless backend so it works on any machine / CI
import matplotlib.pyplot as plt
import pandas as pd

from src import config

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(config.ROOT)}")


def run_eda():
    config.ensure_dirs()
    df = pd.read_csv(config.RAW_DATA)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols from {config.RAW_DATA.name}")

    # --- 1. Target distribution -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[config.TARGET], bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(config.RISK_THRESHOLD, color="#C44E52", linestyle="--",
               label=f"At-Risk threshold (<{config.RISK_THRESHOLD:.0f})")
    ax.set(title="Distribution of Performance Score (Average_Score)",
           xlabel="Performance Score", ylabel="Number of students")
    ax.legend()
    _save(fig, "01_target_distribution.png")

    # --- 2. Correlation of numeric behaviour vs target --------------------- #
    corr = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()[config.TARGET]
    corr = corr.drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set(title="Correlation of behavioural features with Performance Score",
           xlabel="Pearson correlation")
    _save(fig, "02_feature_correlation.png")

    # --- 3. Study hours vs performance (strongest driver) ------------------ #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["Study_Hours_Per_Day"], df[config.TARGET], s=12,
               alpha=0.4, color="#4C72B0")
    ax.set(title="Study Hours per Day vs Performance Score",
           xlabel="Study Hours per Day", ylabel="Performance Score")
    _save(fig, "03_study_vs_score.png")

    # --- 4. Stress vs performance ------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["Stress_Level"], df[config.TARGET], s=12,
               alpha=0.4, color="#C44E52")
    ax.set(title="Stress Level vs Performance Score",
           xlabel="Stress Level (1-10)", ylabel="Performance Score")
    _save(fig, "04_stress_vs_score.png")

    # --- 5. Part-time job impact on study hours (the dataset trade-off) ---- #
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [df.loc[df["Part_Time_Job"] == v, "Study_Hours_Per_Day"]
              for v in ["No", "Yes"]]
    ax.boxplot(groups)
    ax.set_xticklabels(["No job", "Part-time job"])
    ax.set(title="Part-time job vs Study Hours per Day",
           ylabel="Study Hours per Day")
    _save(fig, "05_job_vs_study.png")

    # --- 6. Categorical summary: mean score by major ----------------------- #
    fig, ax = plt.subplots(figsize=(7, 4))
    by_major = df.groupby("Major")[config.TARGET].mean().sort_values()
    ax.bar(by_major.index, by_major.values, color="#8172B3")
    ax.set(title="Average Performance Score by Major", ylabel="Mean score")
    _save(fig, "06_score_by_major.png")

    # --- Text summary used in slides / README ------------------------------ #
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "target": config.TARGET,
        "target_mean": round(float(df[config.TARGET].mean()), 2),
        "target_min": float(df[config.TARGET].min()),
        "target_max": float(df[config.TARGET].max()),
        "at_risk_count": int((df[config.TARGET] < config.RISK_THRESHOLD).sum()),
        "at_risk_pct": round(float((df[config.TARGET] < config.RISK_THRESHOLD).mean() * 100), 1),
        "top_positive_driver": corr.idxmax(),
        "top_negative_driver": corr.idxmin(),
    }
    out = config.REPORTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  saved {out.relative_to(config.ROOT)}")
    print("EDA complete. Key facts:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()

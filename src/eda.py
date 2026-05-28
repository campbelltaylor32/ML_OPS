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

    # convenience: boolean at-risk flag reused by several charts below
    at_risk = df[config.TARGET] < config.RISK_THRESHOLD

    # --- 1. Target distribution -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[config.TARGET], bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(
        config.RISK_THRESHOLD,
        color="#C44E52",
        linestyle="--",
        label=f"At-Risk threshold (<{config.RISK_THRESHOLD:.0f})",
    )
    ax.set(
        title="Distribution of Performance Score (Average_Score)",
        xlabel="Performance Score",
        ylabel="Number of students",
    )
    ax.legend()
    _save(fig, "01_target_distribution.png")

    # --- 2. Correlation of numeric behaviour vs target --------------------- #
    corr = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()[config.TARGET]
    corr = corr.drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set(
        title="Correlation of behavioural features with Performance Score",
        xlabel="Pearson correlation",
    )
    _save(fig, "02_feature_correlation.png")

    # --- 3. Study hours vs performance (strongest driver) ------------------ #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["Study_Hours_Per_Day"], df[config.TARGET], s=12, alpha=0.4, color="#4C72B0")
    ax.set(
        title="Study Hours per Day vs Performance Score",
        xlabel="Study Hours per Day",
        ylabel="Performance Score",
    )
    _save(fig, "03_study_vs_score.png")

    # --- 4. Stress vs performance ------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["Stress_Level"], df[config.TARGET], s=12, alpha=0.4, color="#C44E52")
    ax.set(
        title="Stress Level vs Performance Score",
        xlabel="Stress Level (1-10)",
        ylabel="Performance Score",
    )
    _save(fig, "04_stress_vs_score.png")

    # --- 5. Part-time job impact on study hours (the dataset trade-off) ---- #
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [df.loc[df["Part_Time_Job"] == v, "Study_Hours_Per_Day"] for v in ["No", "Yes"]]
    ax.boxplot(groups)
    ax.set_xticklabels(["No job", "Part-time job"])
    ax.set(title="Part-time job vs Study Hours per Day", ylabel="Study Hours per Day")
    _save(fig, "05_job_vs_study.png")

    # --- 6. Categorical summary: mean score by major ----------------------- #
    fig, ax = plt.subplots(figsize=(7, 4))
    by_major = df.groupby("Major")[config.TARGET].mean().sort_values()
    ax.bar(by_major.index, by_major.values, color="#8172B3")
    ax.set(title="Average Performance Score by Major", ylabel="Mean score")
    _save(fig, "06_score_by_major.png")

    # --- 7. Numeric feature correlation matrix (multicollinearity check) ---- #
    # MLOps note: we keep an eye on feature-to-feature correlation so we don't
    # feed near-duplicate signals into the model during feature selection.
    cmat = df[config.NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cmat.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cmat.columns)))
    ax.set_yticks(range(len(cmat.index)))
    ax.set_xticklabels(cmat.columns, rotation=45, ha="right")
    ax.set_yticklabels(cmat.index)
    ax.grid(False)  # a grid on top of the cells just adds noise
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(
                j,
                i,
                f"{cmat.iat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(cmat.iat[i, j]) > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set(title="Correlation between numeric features")
    _save(fig, "07_feature_correlation_matrix.png")

    # --- 8. At-risk rate by major (who needs help, not just mean score) ---- #
    risk_by_major = df.assign(_risk=at_risk).groupby("Major")["_risk"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(risk_by_major.index, risk_by_major.values, color="#C44E52")
    ax.set(
        title=f"At-Risk rate by Major (score < {config.RISK_THRESHOLD:.0f})",
        ylabel="At-Risk students (%)",
    )
    _save(fig, "08_risk_by_major.png")

    # --- 9. Study hours vs score, split by at-risk flag -------------------- #
    # Reinforces the threshold decision: shows where the at-risk band sits.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df.loc[~at_risk, "Study_Hours_Per_Day"],
        df.loc[~at_risk, config.TARGET],
        s=12,
        alpha=0.4,
        color="#4C72B0",
        label="On track",
    )
    ax.scatter(
        df.loc[at_risk, "Study_Hours_Per_Day"],
        df.loc[at_risk, config.TARGET],
        s=12,
        alpha=0.6,
        color="#C44E52",
        label="At-risk",
    )
    ax.axhline(
        config.RISK_THRESHOLD,
        color="#C44E52",
        linestyle="--",
        label=f"At-Risk threshold (<{config.RISK_THRESHOLD:.0f})",
    )
    ax.set(
        title="Study Hours vs Performance Score (by risk status)",
        xlabel="Study Hours per Day",
        ylabel="Performance Score",
    )
    ax.legend()
    _save(fig, "09_study_vs_score_by_risk.png")

    # --- 10. Part-time job vs performance (close the trade-off story) ------- #
    # Chart 5 showed a job costs study hours; this shows whether that actually
    # reads through to the score the model has to predict.
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [df.loc[df["Part_Time_Job"] == v, config.TARGET] for v in ["No", "Yes"]]
    ax.boxplot(groups)
    ax.set_xticklabels(["No job", "Part-time job"])
    ax.set(title="Part-time job vs Performance Score", ylabel="Performance Score")
    _save(fig, "10_job_vs_score.png")

    # --- 11. Distributions of the numeric behavioural features ------------- #
    feats = config.NUMERIC_FEATURES
    ncols = 3
    nrows = -(-len(feats) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, feat in zip(axes.ravel(), feats):
        ax.hist(df[feat], bins=25, color="#4C72B0", edgecolor="white")
        ax.set(title=feat, ylabel="Count")
    for ax in axes.ravel()[len(feats) :]:  # blank out any unused panels
        ax.set_visible(False)
    fig.suptitle("Distributions of numeric behavioural features", y=1.02)
    _save(fig, "11_numeric_distributions.png")

    # --- Text summary used in slides / README ------------------------------ #
    # strongest off-diagonal correlation among numeric features (collinearity)
    abs_cmat = cmat.abs()
    for f in abs_cmat.columns:
        abs_cmat.loc[f, f] = 0.0
    top_pair_val = abs_cmat.values.max()
    i, j = divmod(abs_cmat.values.argmax(), abs_cmat.shape[1])
    top_pair = sorted([abs_cmat.index[i], abs_cmat.columns[j]])

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "target": config.TARGET,
        "target_mean": round(float(df[config.TARGET].mean()), 2),
        "target_min": float(df[config.TARGET].min()),
        "target_max": float(df[config.TARGET].max()),
        "at_risk_count": int(at_risk.sum()),
        "at_risk_pct": round(float(at_risk.mean() * 100), 1),
        "top_positive_driver": corr.idxmax(),
        "top_negative_driver": corr.idxmin(),
        "highest_risk_major": risk_by_major.idxmax(),
        "highest_risk_major_pct": round(float(risk_by_major.max()), 1),
        "at_risk_pct_part_time": round(
            float(at_risk[df["Part_Time_Job"] == "Yes"].mean() * 100), 1
        ),
        "at_risk_pct_no_job": round(float(at_risk[df["Part_Time_Job"] == "No"].mean() * 100), 1),
        "strongest_feature_pair": top_pair,
        "strongest_feature_pair_corr": round(float(top_pair_val), 2),
    }
    out = config.REPORTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  saved {out.relative_to(config.ROOT)}")
    print("EDA complete. Key facts:", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()

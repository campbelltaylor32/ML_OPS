"""
Stage 2 — Staging → Intermediate
Drops leakage + ID columns, adds 2 derived features.
Writes data/intermediate/int_student_features.parquet
"""
import pandas as pd

from src import config


def run() -> pd.DataFrame:
    df = pd.read_parquet(config.STAGING_PATH)

    drop_cols = config.LEAKAGE_COLUMNS + config.ID_COLUMNS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    study = df["Study_Hours_Per_Day"].clip(lower=0)
    sleep = df["Sleep_Hours_Per_Day"].clip(lower=1e-3)
    df["study_sleep_ratio"] = (study / sleep).round(4)
    df["effective_attendance"] = (
        df["Attendance_Rate"] * (1 - df["Stress_Level"] / 10.0)
    ).round(4)

    config.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.INTERMEDIATE_PATH, index=False)
    print(f"Intermediate: {len(df)} rows, {len(df.columns)} cols "
          f"-> {config.INTERMEDIATE_PATH.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    run()

"""
Stage 1 — Raw → Staging
1:1 with the raw CSV; light hygiene only (no feature dropping, no target removal).
Writes data/staging/stg_students.parquet
"""

import pandas as pd

from src import config


def run() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATA)

    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    for col in config.NUMERIC_FEATURES + [config.TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=config.NUMERIC_FEATURES + [config.TARGET])

    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.STAGING_PATH, index=False)
    print(f"Staging: {len(df)} rows -> {config.STAGING_PATH.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    run()

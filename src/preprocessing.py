"""
preprocessing.py  --  Step 3: clean columns and split train / test
===================================================================
MLOps purpose: DATA PREPARATION + REPRODUCIBLE SPLIT.
We drop leakage/identifier columns, keep only the agreed feature set + target,
and split into train/test with a FIXED SEED so every teammate and every rerun
gets the exact same rows. The split is written to data/processed/ so all
downstream steps (training, evaluation, monitoring) read identical files.

Note: feature scaling / one-hot encoding is NOT done here. It lives INSIDE the
model pipeline (train_automl.py) so that the exact same transformations are
applied automatically at inference time -- a core MLOps principle that prevents
training/serving skew.

Run:  python -m src.preprocessing
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_clean() -> pd.DataFrame:
    """Load raw CSV and keep only features + target (drop leakage + IDs)."""
    df = pd.read_csv(config.RAW_DATA)
    keep = config.FEATURES + [config.TARGET]
    dropped = [c for c in df.columns if c not in keep]
    df = df[keep].copy()

    # Basic hygiene: drop dup rows, coerce numerics, drop impossible rows.
    df = df.drop_duplicates()
    for col in config.NUMERIC_FEATURES + [config.TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=config.NUMERIC_FEATURES + [config.TARGET])

    print(f"Dropped leakage/ID columns: {dropped}")
    print(f"Clean dataset: {df.shape[0]} rows, {len(config.FEATURES)} features")
    return df


def split_and_save():
    config.ensure_dirs()
    df = load_clean()

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED, shuffle=True
    )
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)

    print(f"Train: {len(train_df)} rows -> {config.TRAIN_PATH.relative_to(config.ROOT)}")
    print(f"Test:  {len(test_df)} rows -> {config.TEST_PATH.relative_to(config.ROOT)}")
    return train_df, test_df


if __name__ == "__main__":
    split_and_save()

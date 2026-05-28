"""
Stage 3 — Intermediate → Mart
Produces analytics-ready train and test parquets using the fixed-seed split.
Writes data/marts/student_training.parquet and student_serving.parquet
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def run() -> tuple:
    df = pd.read_parquet(config.INTERMEDIATE_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        shuffle=True,
    )

    config.MARTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.MART_TRAIN_PATH, index=False)
    test_df.to_parquet(config.MART_TEST_PATH, index=False)

    print(f"Mart train: {len(train_df)} rows -> {config.MART_TRAIN_PATH.relative_to(config.ROOT)}")
    print(f"Mart test:  {len(test_df)} rows -> {config.MART_TEST_PATH.relative_to(config.ROOT)}")
    return train_df, test_df


if __name__ == "__main__":
    run()

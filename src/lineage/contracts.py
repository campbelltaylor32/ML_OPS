"""
contracts.py  --  Checking Guards between lineage stages
=========================================================
Each guard raises ValueError on contract violation so the pipeline halts rather
than silently propagating bad data.
"""
from pathlib import Path
from typing import List

import pandas as pd

from src import config


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")


def validate_staging() -> None:
    _require_file(config.STAGING_PATH)
    df = pd.read_parquet(config.STAGING_PATH)
    if len(df) == 0:
        raise ValueError("Staging table is empty")
    missing = [c for c in config.NUMERIC_FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Staging missing columns: {missing}")


def validate_intermediate() -> None:
    _require_file(config.INTERMEDIATE_PATH)
    df = pd.read_parquet(config.INTERMEDIATE_PATH)
    if len(df) == 0:
        raise ValueError("Intermediate table is empty")
    leakage_present = [c for c in config.LEAKAGE_COLUMNS + config.ID_COLUMNS if c in df.columns]
    if leakage_present:
        raise ValueError(f"Leakage/ID columns still present in intermediate: {leakage_present}")
    nulls = df[config.NUMERIC_FEATURES].isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        raise ValueError(f"Nulls in numeric features: {bad.to_dict()}")


def validate_marts() -> None:
    _require_file(config.MART_TRAIN_PATH)
    _require_file(config.MART_TEST_PATH)
    for path in (config.MART_TRAIN_PATH, config.MART_TEST_PATH):
        df = pd.read_parquet(path)
        if len(df) == 0:
            raise ValueError(f"Mart partition is empty: {path}")
        leakage_present = [c for c in config.LEAKAGE_COLUMNS + config.ID_COLUMNS if c in df.columns]
        if leakage_present:
            raise ValueError(f"Leakage cols in mart {path.name}: {leakage_present}")
        if config.TARGET not in df.columns:
            raise ValueError(f"Target column '{config.TARGET}' missing from mart {path.name}")


def validate_feature_store(versions: List[str] = ("v1", "v2")) -> None:
    for v in versions:
        vdir = config.FEATURE_STORE_DIR / v
        for fname in ("train.parquet", "test.parquet", "feature_defs.json"):
            _require_file(vdir / fname)
        train = pd.read_parquet(vdir / "train.parquet")
        mart_train = pd.read_parquet(config.MART_TRAIN_PATH)
        if len(train) != len(mart_train):
            raise ValueError(
                f"Feature store {v} train row count {len(train)} != mart {len(mart_train)}"
            )


def validate_model_ready() -> None:
    _require_file(config.MODEL_PATH)
    _require_file(config.AUTOML_DIR / "benchmark.json")

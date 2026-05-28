"""
feature_store.py  --  Versioned feature sets for the student_performance pipeline
==================================================================================
Two versions:
  v1 — original 14-feature set from config.FEATURES (baseline)
  v2 — engineered set: same numerics + 2 derived features, drops 2 weak categoricals

Run:  python -m src.feature_store
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

FEATURE_VERSIONS: Dict[str, List[str]] = {
    "v1": config.FEATURES,
    "v2": [
        "Age",
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Attendance_Rate",
        "Stress_Level",
        "study_sleep_ratio",
        "effective_attendance",
        "Gender",
        "Major",
        "Living_Area",
        "Parent_Education",
        "Scholarship_Status",
        "Part_Time_Job",
        "Internet_Access",
    ],
}

CATEGORICAL_V2 = [
    "Gender", "Major", "Living_Area", "Parent_Education",
    "Scholarship_Status", "Part_Time_Job", "Internet_Access",
]
NUMERIC_V2 = [
    "Age", "Study_Hours_Per_Day", "Sleep_Hours_Per_Day",
    "Attendance_Rate", "Stress_Level",
    "study_sleep_ratio", "effective_attendance",
]


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    study = out["Study_Hours_Per_Day"].clip(lower=0)
    sleep = out["Sleep_Hours_Per_Day"].clip(lower=1e-3)
    out["study_sleep_ratio"] = (study / sleep).round(4)
    out["effective_attendance"] = (
        out["Attendance_Rate"] * (1 - out["Stress_Level"] / 10.0)
    ).round(4)
    return out


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:12]


class FeatureStore:
    def __init__(self, store_dir: Path = config.FEATURE_STORE_DIR):
        self.store_dir = store_dir

    def _version_dir(self, version: str) -> Path:
        d = self.store_dir / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def materialize(self, mart_train: pd.DataFrame, mart_test: pd.DataFrame) -> None:
        for version, features in FEATURE_VERSIONS.items():
            train = self._prepare(mart_train, version)
            test = self._prepare(mart_test, version)
            vdir = self._version_dir(version)
            train.to_parquet(vdir / "train.parquet", index=False)
            test.to_parquet(vdir / "test.parquet", index=False)
            meta = {
                "version": version,
                "features": features,
                "n_features": len(features),
                "n_train": len(train),
                "n_test": len(test),
                "train_hash": _hash_df(train[features]),
            }
            (vdir / "feature_defs.json").write_text(json.dumps(meta, indent=2))

    def _prepare(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        if version == "v2":
            df = _add_derived(df)
        features = FEATURE_VERSIONS[version]
        cols = [c for c in features if c in df.columns] + [config.TARGET]
        return df[cols].copy()

    def get_feature_set(
        self, version: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
        vdir = self._version_dir(version)
        train = pd.read_parquet(vdir / "train.parquet")
        test = pd.read_parquet(vdir / "test.parquet")
        features = FEATURE_VERSIONS[version]
        meta = json.loads((vdir / "feature_defs.json").read_text())
        present = [f for f in features if f in train.columns]
        return (
            train[present],
            test[present],
            train[config.TARGET],
            test[config.TARGET],
            meta,
        )

    def is_ready(self, version: str) -> bool:
        vdir = self.store_dir / version
        return (
            (vdir / "train.parquet").exists()
            and (vdir / "test.parquet").exists()
            and (vdir / "feature_defs.json").exists()
        )


def build_feature_store() -> None:
    config.ensure_dirs()
    mart_train = pd.read_parquet(config.MART_TRAIN_PATH)
    mart_test = pd.read_parquet(config.MART_TEST_PATH)
    fs = FeatureStore()
    fs.materialize(mart_train, mart_test)
    for version in FEATURE_VERSIONS:
        meta_path = fs.store_dir / version / "feature_defs.json"
        meta = json.loads(meta_path.read_text())
        print(f"Feature store {version}: {meta['n_features']} features, "
              f"{meta['n_train']} train rows")


if __name__ == "__main__":
    build_feature_store()

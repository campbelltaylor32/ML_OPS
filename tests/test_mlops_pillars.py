"""
test_mlops_pillars.py  --  Guard tests G1–G4 for the four MLOps pillars
Run:  pytest -q  (after running run_all.sh with the new enhanced pipeline)
"""

import json

import pandas as pd

from src import config
from src.lineage.contracts import (
    validate_feature_store,
    validate_intermediate,
    validate_marts,
    validate_model_ready,
    validate_staging,
)


def test_g1_staging_contract():
    validate_staging()


def test_g1_intermediate_contract():
    validate_intermediate()


def test_g1_marts_contract():
    validate_marts()


def test_g1_no_leakage_in_marts():
    for path in (config.MART_TRAIN_PATH, config.MART_TEST_PATH):
        df = pd.read_parquet(path)
        leakage = set(config.LEAKAGE_COLUMNS + config.ID_COLUMNS) & set(df.columns)
        assert not leakage, f"Leakage in {path.name}: {leakage}"


def test_g1_derived_features_in_intermediate():
    df = pd.read_parquet(config.INTERMEDIATE_PATH)
    assert "study_sleep_ratio" in df.columns
    assert "effective_attendance" in df.columns


def test_g2_feature_store_all_versions():
    validate_feature_store(versions=["v1", "v2"])


def test_g2_feature_store_v2_has_derived():
    v2_dir = config.FEATURE_STORE_DIR / "v2"
    defs = json.loads((v2_dir / "feature_defs.json").read_text())
    assert "study_sleep_ratio" in defs["features"]
    assert "effective_attendance" in defs["features"]


def test_g2_feature_store_row_count_matches_marts():
    mart_train = pd.read_parquet(config.MART_TRAIN_PATH)
    for v in ("v1", "v2"):
        train = pd.read_parquet(config.FEATURE_STORE_DIR / v / "train.parquet")
        assert len(train) == len(mart_train), f"v{v} train row mismatch"


def test_g3_model_and_benchmark_exist():
    validate_model_ready()


def test_g3_benchmark_has_flaml_entry():
    data = json.loads((config.AUTOML_DIR / "benchmark.json").read_text())
    regimes = {r["regime"] for r in data}
    assert "baseline" in regimes
    assert "flaml" in regimes


def test_g4_monitoring_reference_and_current_exist():
    assert config.TEST_PATH.exists(), "Reference test set missing"
    assert config.TEST_MODIFIED_PATH.exists(), "Modified (current) test set missing"


def test_g4_evidently_report_generated():
    report = config.MONITORING_DIR / "evidently_report.html"
    assert report.exists(), "Evidently HTML report not found"
    assert report.stat().st_size > 1000, "Evidently HTML report appears empty"

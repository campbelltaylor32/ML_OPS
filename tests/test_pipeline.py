"""
test_pipeline.py  --  lightweight smoke tests
==============================================
MLOps purpose: basic CI guardrails. Confirms no leakage columns slip into the
feature set, the split sizes are right, the saved model loads and predicts in
range, and the stress scenario actually lowers predicted scores.

Run:  pytest -q   (after running the pipeline so artifacts exist)
"""
import joblib
import pandas as pd

from src import config
from src.make_modified_test import apply_scenario


def test_no_leakage_in_features():
    assert not set(config.FEATURES) & set(config.LEAKAGE_COLUMNS)
    assert config.TARGET not in config.FEATURES


def test_split_sizes():
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    total = len(train) + len(test)
    assert abs(len(test) / total - config.TEST_SIZE) < 0.02


def test_model_predicts_in_range():
    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.TEST_PATH)
    preds = model.predict(test[config.FEATURES])
    assert preds.min() >= 0 and preds.max() <= 100
    assert len(preds) == len(test)


def test_stress_scenario_lowers_scores():
    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.TEST_PATH)
    modified = apply_scenario(test)
    base = model.predict(test[config.FEATURES]).mean()
    stressed = model.predict(modified[config.FEATURES]).mean()
    assert stressed < base  # stress should reduce predicted performance


def test_at_least_two_features_changed():
    assert len(config.STRESS_SCENARIO) >= 2

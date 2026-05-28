"""
experiment.py  --  Feature-set × hyperparameter grid with CodeCarbon tracking
==============================================================================
Runs one MLflow run per (feature_version, estimator_config) combination.
Each run logs: feature metadata, hyperparams, RMSE/MAE/R2, co2_kg_emissions.

Run:  python -m src.experiment
"""

import json
import time
from typing import Any, Dict, List

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.evaluate import regression_metrics
from src.feature_store import CATEGORICAL_V2, FeatureStore
from src.lineage.contracts import validate_feature_store
from src.mlflow_utils import log_feature_set, start_tracked_run, track_emissions

HYPERPARAM_GRID: List[Dict[str, Any]] = [
    {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 63},
    {"n_estimators": 100, "learning_rate": 0.2, "num_leaves": 15},
]


def _build_pipeline(cat_features: List[str], params: Dict[str, Any]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough",
    )
    model = lgb.LGBMRegressor(
        random_state=config.SEED,
        verbose=-1,
        **params,
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def run_grid() -> List[Dict[str, Any]]:
    config.ensure_dirs()
    validate_feature_store()

    fs = FeatureStore()
    results = []

    for version in ("v1", "v2"):
        X_train, X_test, y_train, y_test, fmeta = fs.get_feature_set(version)
        cat_cols = CATEGORICAL_V2 if version == "v2" else config.CATEGORICAL_FEATURES
        cat_cols = [c for c in cat_cols if c in X_train.columns]

        for hp in HYPERPARAM_GRID:
            run_name = f"lgbm_{version}_lr{hp['learning_rate']}_n{hp['n_estimators']}"
            with start_tracked_run(run_name, tags={"feature_version": version}) as run:
                with track_emissions("student_experiment"):
                    log_feature_set(fmeta)
                    import mlflow

                    mlflow.log_params(hp)

                    pipe = _build_pipeline(cat_cols, hp)
                    t0 = time.perf_counter()
                    pipe.fit(X_train, y_train)
                    train_sec = round(time.perf_counter() - t0, 3)

                    preds = pipe.predict(X_test)
                    metrics = regression_metrics(y_test, preds)
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("train_sec", train_sec)

                row = {
                    "feature_version": version,
                    **hp,
                    **metrics,
                    "train_sec": train_sec,
                    "run_id": run.info.run_id,
                }
                results.append(row)
                print(f"{run_name}: RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    out = config.EXPERIMENTS_DIR / "feature_grid.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nExperiment grid saved -> {out.relative_to(config.ROOT)}")
    return results


if __name__ == "__main__":
    run_grid()

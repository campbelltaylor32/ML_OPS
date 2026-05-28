"""
automl_benchmark.py  --  Baseline vs FLAML vs H2O + latency + top-5 features
==============================================================================
Three regimes, all logged to MLflow:
  1. baseline    — default LightGBM, no tuning
  2. flaml       — FLAML AutoML (reuses train_automl logic)
  3. h2o         — H2O AutoML (JVM-guarded; skips cleanly if unavailable)
Also produces a top-5-feature variant via LightGBM importances.

Run:  python -m src.automl_benchmark
"""
import json
import time
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn

from src import config
from src.evaluate import regression_metrics
from src.feature_store import FeatureStore
from src.lineage.contracts import validate_feature_store
from src.mlflow_utils import start_tracked_run, track_emissions
from src.train_automl import build_preprocessor

LATENCY_ROWS = 200
LATENCY_REPEATS = 30


def _measure_latency(pipeline, X_sample: pd.DataFrame) -> float:
    times = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        pipeline.predict(X_sample)
        times.append(time.perf_counter() - t0)
    return round(float(np.median(times)) * 1000, 3)  # ms


def _ohe_pipeline(cat_features: List[str], model) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough",
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def run_baseline(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    pipe = _ohe_pipeline(
        config.CATEGORICAL_FEATURES,
        lgb.LGBMRegressor(n_estimators=100, random_state=config.SEED, verbose=-1),
    )
    with start_tracked_run("baseline_lgbm", tags={"regime": "baseline"}):
        with track_emissions("automl_baseline"):
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_sec = round(time.perf_counter() - t0, 3)
            preds = pipe.predict(X_test)
            metrics = regression_metrics(y_test, preds)
            lat = _measure_latency(pipe, X_test.head(LATENCY_ROWS))
            mlflow.log_params({"n_estimators": 100, "regime": "baseline"})
            mlflow.log_metrics({**metrics, "train_sec": train_sec, "latency_ms": lat})
    return {"regime": "baseline", **metrics, "train_sec": train_sec,
            "latency_ms": lat, "pipeline": pipe}


def run_flaml(X_train, y_train, X_test, y_test, time_budget: int = 60) -> Dict[str, Any]:
    from flaml import AutoML

    pre = build_preprocessor()
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)

    automl = AutoML()
    with start_tracked_run(f"flaml_automl", tags={"regime": "flaml"}) as run:
        with track_emissions("automl_flaml"):
            t0 = time.perf_counter()
            automl.fit(
                X_train=X_tr, y_train=y_train.values,
                task="regression", metric="rmse",
                time_budget=time_budget,
                estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
                seed=config.SEED, verbose=0,
            )
            train_sec = round(time.perf_counter() - t0, 3)

            final_pipe = Pipeline([("preprocessor", pre),
                                    ("model", automl.model.estimator)])
            preds = final_pipe.predict(X_test)
            metrics = regression_metrics(y_test, preds)
            lat = _measure_latency(final_pipe, X_test.head(LATENCY_ROWS))

            mlflow.log_params({"best_algo": automl.best_estimator, "time_budget": time_budget})
            mlflow.log_metrics({**metrics, "train_sec": train_sec, "latency_ms": lat})
            mlflow.sklearn.log_model(
                sk_model=final_pipe,
                name="model",
                registered_model_name=config.REGISTERED_MODEL_NAME,
                input_example=X_train.head(2),
            )
            import joblib
            joblib.dump(final_pipe, config.MODEL_PATH)

    return {"regime": "flaml", "best_algo": automl.best_estimator,
            **metrics, "train_sec": train_sec, "latency_ms": lat}


def run_h2o(X_train, y_train, X_test, y_test, max_models: int = 5) -> Dict[str, Any]:
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError:
        print("H2O not installed — skipping H2O benchmark.")
        return {"regime": "h2o", "status": "skipped", "reason": "h2o not installed"}

    import logging
    for _log in ("py4j", "requests", "urllib3", "h2o"):
        logging.getLogger(_log).setLevel(logging.ERROR)

    try:
        h2o.init(nthreads=-1, max_mem_size="2G", verbose=False, log_level="ERRR")
        h2o.no_progress()  # disable per-call progress bars globally for this session

        train_df = X_train.copy()
        train_df[config.TARGET] = y_train.values
        test_df = X_test.copy()
        test_df[config.TARGET] = y_test.values

        h2o_train = h2o.H2OFrame(train_df)
        h2o_test = h2o.H2OFrame(test_df)
        h2o_latency_frame = h2o.H2OFrame(X_test.head(LATENCY_ROWS))

        with start_tracked_run("h2o_automl", tags={"regime": "h2o"}):
            with track_emissions("automl_h2o"):
                t0 = time.perf_counter()
                aml = H2OAutoML(
                    max_models=max_models,
                    seed=config.SEED,
                    nfolds=3,
                    verbosity=None,
                )
                aml.train(y=config.TARGET, training_frame=h2o_train)
                train_sec = round(time.perf_counter() - t0, 3)

                perf = aml.leader.model_performance(h2o_test)
                metrics = {
                    "RMSE": round(float(perf.rmse()), 6),
                    "MAE": round(float(perf.mae()), 6),
                    "R2": round(float(perf.r2()), 6),
                }

                # H2O latency: warm up the JVM bridge once, then take 10 timed samples
                # (30 repeats × stacked-ensemble overhead stresses the connection)
                h2o_repeats = 10
                aml.leader.predict(h2o_latency_frame)  # warm-up
                times = []
                for _ in range(h2o_repeats):
                    t1 = time.perf_counter()
                    aml.leader.predict(h2o_latency_frame)
                    times.append(time.perf_counter() - t1)
                lat = round(float(np.median(times)) * 1000, 3)

                leader_id = aml.leader.model_id  # capture before shutdown closes the connection
                mlflow.log_params({"max_models": max_models, "leader": leader_id})
                mlflow.log_metrics({**metrics, "train_sec": train_sec, "latency_ms": lat})

        h2o.cluster().shutdown()
        return {"regime": "h2o", "leader": leader_id,
                **metrics, "train_sec": train_sec, "latency_ms": lat}
    except Exception as exc:
        print(f"H2O benchmark failed: {exc}")
        try:
            h2o.cluster().shutdown()
        except Exception:
            pass
        return {"regime": "h2o", "status": "failed", "reason": str(exc)}


def run_top5_variant(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    base_pipe = _ohe_pipeline(
        config.CATEGORICAL_FEATURES,
        lgb.LGBMRegressor(n_estimators=200, random_state=config.SEED, verbose=-1),
    )
    base_pipe.fit(X_train, y_train)

    pre = base_pipe.named_steps["preprocessor"]
    raw_imp = base_pipe.named_steps["model"].feature_importances_

    ohe_cats = pre.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    all_feature_names = list(ohe_cats) + config.NUMERIC_FEATURES
    imp_series = pd.Series(raw_imp[:len(all_feature_names)], index=all_feature_names)

    orig_imp = {}
    for feat in config.FEATURES:
        cols = [c for c in imp_series.index if c == feat or c.startswith(feat + "_")]
        orig_imp[feat] = float(imp_series[cols].sum()) if cols else 0.0

    top5 = sorted(orig_imp, key=orig_imp.get, reverse=True)[:5]
    top5_num = [f for f in top5 if f in config.NUMERIC_FEATURES]
    top5_cat = [f for f in top5 if f in config.CATEGORICAL_FEATURES]

    top5_pipe = _ohe_pipeline(
        top5_cat,
        lgb.LGBMRegressor(n_estimators=200, random_state=config.SEED, verbose=-1),
    )
    with start_tracked_run("top5_features", tags={"regime": "top5"}):
        with track_emissions("automl_top5"):
            t0 = time.perf_counter()
            top5_pipe.fit(X_train[top5], y_train)
            train_sec = round(time.perf_counter() - t0, 3)
            preds = top5_pipe.predict(X_test[top5])
            metrics = regression_metrics(y_test, preds)
            lat = _measure_latency(top5_pipe, X_test[top5].head(LATENCY_ROWS))
            mlflow.log_params({"top5_features": ",".join(top5), "n_features": 5})
            mlflow.log_metrics({**metrics, "train_sec": train_sec, "latency_ms": lat})

    return {"regime": "top5", "top5_features": top5,
            **metrics, "train_sec": train_sec, "latency_ms": lat}


def main(flaml_budget: int = 60, h2o_models: int = 5) -> None:
    config.ensure_dirs()
    validate_feature_store()

    fs = FeatureStore()
    X_train, X_test, y_train, y_test, _ = fs.get_feature_set("v1")

    results = []

    print("=== Baseline LightGBM ===")
    res = run_baseline(X_train, y_train, X_test, y_test)
    results.append({k: v for k, v in res.items() if k != "pipeline"})

    print("=== FLAML AutoML ===")
    res = run_flaml(X_train, y_train, X_test, y_test, time_budget=flaml_budget)
    results.append(res)

    print("=== H2O AutoML ===")
    res = run_h2o(X_train, y_train, X_test, y_test, max_models=h2o_models)
    results.append(res)

    print("=== Top-5 Feature Variant ===")
    res = run_top5_variant(X_train, y_train, X_test, y_test)
    results.append(res)

    config.AUTOML_DIR.mkdir(parents=True, exist_ok=True)
    out = config.AUTOML_DIR / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nBenchmark results -> {out.relative_to(config.ROOT)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scored = [r for r in results if "RMSE" in r and "latency_ms" in r]
        labels = [r["regime"] for r in scored]
        rmses = [r["RMSE"] for r in scored]
        lats = [r["latency_ms"] for r in scored]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(lats, rmses, s=80, zorder=5)
        for label, x, y in zip(labels, lats, rmses):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=9)
        ax.set(xlabel="Median Latency (ms, 200 rows)",
               ylabel="Test RMSE",
               title="AutoML Accuracy vs Inference Speed")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(config.AUTOML_DIR / "accuracy_vs_latency.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Chart -> {(config.AUTOML_DIR / 'accuracy_vs_latency.png').relative_to(config.ROOT)}")
    except Exception as exc:
        print(f"Chart skipped: {exc}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--flaml-budget", type=int, default=60)
    ap.add_argument("--h2o-models", type=int, default=5)
    args = ap.parse_args()
    main(flaml_budget=args.flaml_budget, h2o_models=args.h2o_models)

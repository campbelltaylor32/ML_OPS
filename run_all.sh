#!/usr/bin/env bash
# run_all.sh -- run the entire enhanced MLOps pipeline end to end.
# Pillar order: Data Lineage -> Feature Store -> Experiments -> AutoML -> Evaluate -> Monitor
set -e

echo "[1/11] EDA .............................."; python -m src.eda
echo "[2/11] Data Lineage (Raw→Mart) .........."; python -m src.lineage.run_lineage
echo "[3/11] Preprocess + split (CSV compat) .."; python -m src.preprocessing
echo "[4/11] Feature Store (v1 + v2) .........."; python -m src.feature_store
echo "[5/11] Experiment grid (feat × hparams) ."; python -m src.experiment
echo "[6/11] AutoML Benchmark (FLAML+H2O) ....."; python -m src.automl_benchmark --flaml-budget 60
echo "[7/11] Evaluate (original test) ........."; python -m src.evaluate
echo "[8/11] Make modified test set ..........."; python -m src.make_modified_test
echo "[9/11] Batch inference + compare ........"; python -m src.batch_inference
echo "[10/11] PSI drift monitoring ............."; python -m src.monitoring
echo "[11/11] Evidently drift report ..........."; python -m src.monitoring_evidently

echo ""
echo "Done. Next:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking + registry"
echo "  streamlit run app/dashboard.py                      # inference + monitoring dashboard"
echo "  uvicorn app.inference_api:app --port 8000           # REST API (http://localhost:8000/docs)"
echo "  pytest -q                                           # run all guard tests (G1-G4)"

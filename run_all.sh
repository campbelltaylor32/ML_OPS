#!/usr/bin/env bash
# run_all.sh -- run the entire MLOps pipeline end to end.
# MLOps purpose: REPRODUCIBLE ORCHESTRATION. One command regenerates every
# artifact (charts, split, model, metrics, modified set, drift report) from the
# raw CSV, so anyone can reproduce the results before the demo.
set -e

echo "[1/7] EDA ..............................."; python -m src.eda
echo "[2/7] Preprocess + split ..............."; python -m src.preprocessing
echo "[3/7] AutoML train (MLflow) ............"; python -m src.train_automl --time-budget 60
echo "[4/7] Evaluate (original test) ........."; python -m src.evaluate
echo "[5/7] Make modified test set ..........."; python -m src.make_modified_test
echo "[6/7] Batch inference + compare ........"; python -m src.batch_inference
echo "[7/7] Drift monitoring ................."; python -m src.monitoring

echo ""
echo "Done. Next:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment tracking + registry"
echo "  streamlit run app/dashboard.py                      # inference + monitoring dashboard"
echo "  uvicorn app.inference_api:app --port 8000           # REST API (http://localhost:8000/docs)"

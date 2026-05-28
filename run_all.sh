#!/usr/bin/env bash
# run_all.sh -- run the entire enhanced MLOps pipeline end to end.
# Pillar order: Data Lineage -> Feature Store -> Experiments -> AutoML -> Evaluate -> Monitor
set -e

echo "[1/11] EDA .............................."; uv run python -m src.eda
echo "[2/11] Data Lineage (Raw→Mart) .........."; uv run python -m src.lineage.run_lineage
echo "[3/11] Preprocess + split (CSV compat) .."; uv run python -m src.preprocessing
echo "[4/11] Feature Store (v1 + v2) .........."; uv run python -m src.feature_store
echo "[5/11] Experiment grid (feat × hparams) ."; uv run python -m src.experiment
echo "[6/11] AutoML Benchmark (FLAML) .........."; uv run python -m src.automl_benchmark --flaml-budget 60
echo "[7/11] Evaluate (original test) ........."; uv run python -m src.evaluate
echo "[8/11] Make modified test set ..........."; uv run python -m src.make_modified_test
echo "[9/11] Batch inference + compare ........"; uv run python -m src.batch_inference
echo "[10/11] PSI drift monitoring ............."; uv run python -m src.monitoring
echo "[11/11] Evidently drift report ..........."; uv run python -m src.monitoring_evidently

echo ""
echo "Done. Next:"
echo "  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # http://localhost:5001"
echo "  uv run streamlit run app/dashboard.py                      # inference + monitoring dashboard"
echo "  uv run uvicorn app.inference_api:app --port 8000           # REST API (http://localhost:8000/docs)"
echo "  uv run python -m pytest -q                                 # run all guard tests (G1-G4)"
echo ""
echo "  OR: docker compose up   (serves all three apps in containers)"

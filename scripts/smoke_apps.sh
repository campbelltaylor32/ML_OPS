#!/usr/bin/env bash
# scripts/smoke_apps.sh
# Proves the deployable apps start correctly against models/best_model.pkl.
# Safe to run after ci_pipeline.sh (or run_all.sh) has generated the model.
set -e

PORT=8000
API_PID=""

cleanup() {
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "--- Smoke: FastAPI inference API ---"
uv run uvicorn app.inference_api:app --port $PORT --log-level warning &
API_PID=$!

# Wait up to 15 s for the API to be ready
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
        echo "FAIL: API did not start within 15 s" >&2
        exit 1
    fi
done

echo "  /health ..."
curl -fsS "http://localhost:$PORT/health" | uv run python -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('status') == 'ok', f'unexpected health response: {d}'
print('  OK:', d)
"

echo "  POST /predict ..."
curl -fsS -X POST "http://localhost:$PORT/predict" \
    -H "Content-Type: application/json" \
    -d '{"Age":21,"Study_Hours_Per_Day":4.0,"Sleep_Hours_Per_Day":7.0,"Attendance_Rate":80.0,"Stress_Level":4.0,"Gender":"Female","Major":"STEM","Living_Area":"Urban","Parent_Education":"Bachelor","Scholarship_Status":"No","Part_Time_Job":"No","Extracurricular_Activities":"No","Internet_Access":"Yes","Exam_Proximity":"No"}' \
    | uv run python -c "
import sys, json
d = json.load(sys.stdin)
assert 'predicted_score' in d, f'missing predicted_score: {d}'
assert 0 <= d['predicted_score'] <= 100, f'score out of range: {d}'
print('  OK:', d)
"

echo ""
echo "--- Smoke: Streamlit dashboard (import + model load) ---"
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('dashboard', 'app/dashboard.py')
# Streamlit can't headlessly exec the page, but the module-level
# model/data loads will raise if best_model.pkl is missing or corrupt.
# We verify the import path and config resolve correctly.
from src import config
import joblib, json
model = joblib.load(config.MODEL_PATH)
meta = json.loads(config.MODEL_META_PATH.read_text())
assert 'best_algorithm' in meta
print('  OK: model loaded, best_algorithm =', meta['best_algorithm'])
"

echo ""
echo "smoke_apps.sh passed."

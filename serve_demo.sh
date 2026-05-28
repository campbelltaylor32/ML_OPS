#!/usr/bin/env bash
# serve_demo.sh
# Launch all three live apps and print every URL this project serves.
# Press Ctrl-C to stop everything cleanly.
#
# Optional ngrok tunnels: set NGROK_AUTHTOKEN in your shell before running.
#   export NGROK_AUTHTOKEN=<your-token>
#   bash serve_demo.sh

set -e

PIDS=()

cleanup() {
    echo ""
    echo "Stopping all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Kill ngrok if running
    pkill -f "ngrok http" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "Starting services..."

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 \
    --host 127.0.0.1 >/dev/null 2>&1 &
PIDS+=($!)

uvicorn app.inference_api:app --port 8000 --host 127.0.0.1 \
    --log-level warning >/dev/null 2>&1 &
PIDS+=($!)

streamlit run app/dashboard.py --server.port 8501 \
    --server.headless true >/dev/null 2>&1 &
PIDS+=($!)

# Wait for FastAPI to be ready before printing URLs
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# ── Optional ngrok tunnels ────────────────────────────────────────────────────
STREAMLIT_PUBLIC=""
API_PUBLIC=""
if [ -n "$NGROK_AUTHTOKEN" ] && command -v ngrok >/dev/null 2>&1; then
    echo "Opening ngrok tunnels..."
    ngrok http 8501 --log=false >/tmp/ngrok_streamlit.log 2>&1 &
    ngrok http 8000 --log=false >/tmp/ngrok_api.log 2>&1 &
    sleep 3
    STREAMLIT_PUBLIC=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
        | python -c "import sys,json; ts=json.load(sys.stdin)['tunnels']; \
          [print(t['public_url']) for t in ts if '8501' in t.get('config',{}).get('addr','')]" \
        2>/dev/null || true)
    API_PUBLIC=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
        | python -c "import sys,json; ts=json.load(sys.stdin)['tunnels']; \
          [print(t['public_url']) for t in ts if '8000' in t.get('config',{}).get('addr','')]" \
        2>/dev/null || true)
fi

# ── URL Index ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STUDENT PERFORMANCE RISK SYSTEM — ALL SERVICES"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  LIVE APPS (local)"
echo "  ─────────────────────────────────────────────────────────────"
echo "  Streamlit dashboard  (Predict + Monitoring)"
echo "    → http://localhost:8501"
echo ""
echo "  FastAPI inference API  (Swagger UI)"
echo "    → http://localhost:8000/docs"
echo "  FastAPI health probe"
echo "    → http://localhost:8000/health"
echo ""
echo "  MLflow UI  (experiments + model registry)"
echo "    → http://localhost:5000"
echo ""
if [ -n "$STREAMLIT_PUBLIC" ] || [ -n "$API_PUBLIC" ]; then
echo "  NGROK PUBLIC TUNNELS"
echo "  ─────────────────────────────────────────────────────────────"
[ -n "$STREAMLIT_PUBLIC" ] && echo "  Streamlit  → $STREAMLIT_PUBLIC"
[ -n "$API_PUBLIC"        ] && echo "  FastAPI    → $API_PUBLIC/docs"
echo ""
fi
echo "  STATIC REPORTS  (file:// — open in browser)"
echo "  ─────────────────────────────────────────────────────────────"
echo "  Evidently drift report"
echo "    → $(pwd)/reports/monitoring/evidently_report.html"
echo "  AutoML accuracy-vs-latency chart"
echo "    → $(pwd)/reports/automl/accuracy_vs_latency.png"
echo "  Lineage DAG"
echo "    → $(pwd)/reports/lineage/lineage_dag.png"
echo "  All EDA figures"
echo "    → $(pwd)/reports/figures/"
echo ""
echo "  URL INDEX PAGE  (open in browser for one-click access)"
echo "    → $(pwd)/app/serve_index.html"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ctrl-C to stop all services"
echo "════════════════════════════════════════════════════════════════"
echo ""

wait

# E2E Audit Process Log — 2026-05-28

## Phase 1: Audit Findings

### Files to clean up
- `powermetrics_log.txt` (31KB) — macOS system monitoring log; not in .gitignore; not needed → DELETE + gitignore
- `mlflow.db` (13MB) — already gitignored; local tracking store; leave in place

### Untracked files to commit
- `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `uv.lock`, `.dockerignore`, `.python-version`
- These are the Docker / uv infra added in last session — must be tracked to reproduce the Docker E2E

### CI/CD mismatches (critical)
1. `.github/workflows/ci.yml` — uses **Python 3.11** but Dockerfile + `.python-version` both pin **3.12**
2. CI uses `pip install -r requirements.lock` — Docker uses `uv sync --frozen`; CI must match
3. CI caches `~/.cache/pip` keyed on `requirements.lock` — after migrating to uv, cache `~/.cache/uv` keyed on `uv.lock`
4. `scripts/ci_pipeline.sh` uses bare `python -m ...` — needs `uv run` after CI switches to uv
5. `scripts/smoke_apps.sh` uses bare `uvicorn` / `python` — needs `uv run`
6. README shows MLflow URL as `localhost:5000` — should be **`localhost:5001`** (matches serve_demo.sh + docker-compose.yml)

### README setup instructions
- Step 1 shows `python -m venv .venv && pip install -r requirements.lock` — should show `uv sync`
- Step 3 shows `.venv/bin/python -m pytest -q` — should show `uv run python -m pytest -q`

---

## Phase 1: Actions Taken

- [x] Created `process_log.md`
- [x] Added `powermetrics_log.txt` to `.gitignore`
- [x] Deleted `powermetrics_log.txt`
- [x] Fixed `.github/workflows/ci.yml` — Python 3.12 + uv + `uv run` throughout
- [x] Fixed `scripts/ci_pipeline.sh` — bare python → `uv run python`
- [x] Fixed `scripts/smoke_apps.sh` — bare uvicorn/python → `uv run`
- [x] Fixed README — MLflow port 5000 → 5001; setup instructions → uv workflow

## Phase 2: Environment Reset & Docker E2E

- [x] Delete .venv — clean slate confirmed
- [x] Build Docker image — `student-perf-risk:latest` (all layers cached except final COPY)
- [x] Start serving stack — `docker compose up -d mlflow api dashboard`; all 3 containers started + API reported healthy
- [x] `uv sync --frozen` — .venv recreated successfully from uv.lock

## Phase 3: Automated Validation

- [x] `GET /health` → 200 `{"status":"ok","model":"xgboost","registered_model":"student_performance_model"}`
- [x] `POST /predict` → `{"predicted_score":73.52,"at_risk":false,"risk_threshold":60.0}`
- [x] Streamlit (8501) → 200
- [x] MLflow (5001) → 200
- [x] All 17 guard tests (G1–G4) → passed in 48.78s

## Phase 4: CI/CD Alignment — DONE (in Phase 1 fixes)

- [x] Python version: 3.11 → 3.12 (matches Dockerfile)
- [x] Installer: pip + requirements.lock → uv + uv.lock
- [x] Test runner: bare pytest → `uv run python -m pytest -q`
- [x] Pipeline script: bare python → uv run
- [x] Smoke script: bare uvicorn/python → uv run

## Phase 5: Final Report — COMPLETE (see below)

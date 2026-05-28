# TASKS.md — ML_OPS / student_performance_risk

Persistent task tracker. Update at session start and end.
Format: `[ ]` open · `[~]` in-progress · `[x]` done

---

## Infrastructure

- [x] Initialize four-pillar pipeline (lineage, feature store, AutoML, monitoring) — 2026-05-28
- [x] Pin all dependencies (`requirements.txt` + `requirements.lock`) — 2026-05-28
- [x] Fix evidently 0.7.x import paths in `monitoring_evidently.py` — 2026-05-28
- [x] Initialize DVC (`dvc init`, `dvc.yaml`, `params.yaml`, `.dvcignore`) — 2026-05-28
- [x] Add Data & Versioning + Task & Planning Protocol sections to CLAUDE.md — 2026-05-28

## Integration

- [x] Add `_get_dvc_hash` helper to `src/mlflow_utils.py` and call it in `automl_benchmark.py`
  — Every AutoML run should log `dvc_data_hash` param to MLflow for traceability — 2026-05-28

## Pipeline / Model

- [x] Raw CSV kept in git (100KB, self-contained reference); DVC manages pipeline deps only — `dvc add` not needed — 2026-05-28
- [x] Validate `dvc repro` runs clean end-to-end (lineage → feature_store → automl → monitoring) — 2026-05-28
- [x] Confirm `dvc.lock` committed after first successful repro — 2026-05-28

## Serving / App

- [x] Verify `app/dashboard.py` and `app/inference_api.py` load `models/best_model.pkl` correctly after DVC repro — 2026-05-28

## Monitoring

- [x] Confirm Evidently drift report regenerates via `dvc repro monitoring_evidently` after raw data update — 2026-05-28

---

## CI/CD & Demo Serving

- [x] Create `scripts/ci_pipeline.sh` — reduced 8-step pipeline for CI (FLAML budget 15s, skips H2O/EDA/experiment) — 2026-05-28
- [x] Create `scripts/smoke_apps.sh` — boots FastAPI, asserts `/health` + `/predict`, import-checks dashboard — 2026-05-28
- [x] Create `.github/workflows/ci.yml` — push/PR triggered: install → pipeline → pytest → smoke — 2026-05-28
- [x] Create `serve_demo.sh` — launches all 3 apps + prints full URL index; optional ngrok tunnels if `NGROK_AUTHTOKEN` set — 2026-05-28
- [x] Create `app/serve_index.html` — static landing page with clickable links to all apps + reports — 2026-05-28
- [x] Add CI badge + "Live Services & URLs" table to README.md — 2026-05-28

## Backlog

- [x] Add `dvc metrics` tracking for accuracy, F1, PSI scores — wired via `metrics: cache: false` in dvc.yaml for evaluate, batch_inference, monitoring stages — 2026-05-28
- [ ] Integrate DVC pipeline status badge into README

## Goal A — Repo Cleanup Audit — 2026-05-28

- [x] Audit all tracked files vs .gitignore rules
- [x] Remove `reports/experiments/feature_grid.json` from git (DVC output, now gitignored)
- [x] Remove `reports/lineage/lineage_dag.md` + `.png` from git (generated, now gitignored)
- [x] Delete `process_log.md` (session artifact per CLAUDE.md protocol)
- [x] Update `reports/.gitignore` — add `/experiments` and `/lineage`
- [x] Commit pending changes: dvc.yaml (9-stage pipeline), docker-compose.yml, .dvcignore, CLAUDE.md, README.md
- [x] Confirm no dead scripts (all src/ modules referenced in dvc.yaml or documented commands)

## Goal B — Docker E2E Verification — 2026-05-28

- [x] Run `docker compose run --rm pipeline` — all 11 steps complete cleanly
- [x] Confirm artifacts on host via bind-mount: `models/best_model.pkl` (660K), `reports/monitoring/evidently_report.html` (4.0M), `mlflow.db` (14M)
- [x] Confirm Evidently Cloud upload — uploaded to project 019e6f0a-1dec-74b5-9736-1409c8d5b681
- [x] Run `docker compose up` — MLflow (5001) 200 ✓, FastAPI (8000) healthy ✓, Streamlit (8501) 200 ✓
- [x] Run `uv run python -m pytest -q` — 17/17 passed in 10.33s
- [x] Run `bash scripts/smoke_apps.sh` — `/health` + `/predict` assertions pass ✓
- [x] Fix Dockerfile: add `libgomp1` (LightGBM OpenMP runtime missing in slim image)
- [x] Fix `dvc.yaml`: all stage cmds → `uv run python -m src.*` (bare `python` missed venv)
- [x] Run `dvc repro` — 9 stages reproduced cleanly; `dvc.lock` updated
- [x] Run `dvc metrics show` — RMSE=4.08, R²=0.85, risk_F1=0.85, PSI=3.05 (major drift on 4 features)

---

## Audit Findings — 2026-05-28 (gaps to close before final submission)

### P0 — Rubric-blocking
- [x] **Add team member responsibilities section to README** — added table with all four members and per-pillar contributions — 2026-05-28

### P1 — DVC correctness / completeness
- [x] **Add `outs:` to `monitoring_evidently` stage in `dvc.yaml`** — stage now tracks `reports/monitoring/evidently_report.html` — 2026-05-28
- [x] **Extend `dvc.yaml` to cover the full 11-step pipeline** — added `preprocessing`, `experiment`, `evaluate`, `make_modified_test`, `batch_inference`, `monitoring` stages; full pipeline now reproducible via `dvc repro` — 2026-05-28

### P2 — Best practice
- [x] Wire `dvc metrics` for RMSE, F1, and PSI — `evaluate` → `reports/eval_original.json`, `batch_inference` → `reports/metrics_comparison.json`, `monitoring` → `reports/monitoring/drift_report.json` all wired as `metrics: cache: false` — 2026-05-28

---

## Upcoming Goals (run `/goal` to activate)

### Goal A — Repo cleanup audit
Audit every file and document in the repo. Identify and remove anything not part of the core reproducible pipeline (dead scripts, unused notebooks, redundant configs, stale generated files committed by mistake). Clean github history if needed.

### Goal B — End-to-end verification (Docker-first)
Run the full pipeline via Docker (`docker compose run --rm pipeline`), confirm all artifacts are written to the host via bind-mount, start the serving stack (`docker compose up`), run guard tests (`uv run python -m pytest -q`), and smoke tests (`bash scripts/smoke_apps.sh`). Verify Evidently Cloud upload succeeds (token in `.env`). Confirm `dvc repro` runs cleanly and `dvc metrics show` outputs RMSE/F1/PSI. Report pass/fail on every step. Docker is the canonical verification path — `bash run_all.sh` is a fallback only.

---

## Audit Close-Out — 2026-05-28

**All rubric objectives and MLOps best practices met. No further tasks required.**

All 10 rubric items verified and confirmed compliant:
1. ✅ Dataset with outcome variable (`Average_Score`, 1 000 records)
2. ✅ Train/test split (fixed-seed, contract-validated)
3. ✅ Metrics defined (RMSE, MAE, R², risk_accuracy, risk_F1, PSI)
4. ✅ AutoML pipeline (FLAML + H2O + Baseline + Top-5, MLflow-tracked, DVC-managed)
5. ✅ Model deployed (FastAPI + Streamlit + Docker Compose)
6. ✅ Monitoring + dashboard (PSI + Evidently Cloud + Streamlit Monitoring tab)
7. ✅ Original test data validated against deployed model
8. ✅ Modified test set script (`make_modified_test.py`, 4 features changed)
9. ✅ Modified data run against model; delta metrics + PSI drift verified in monitoring
10. ✅ Presentation structure complete including team member responsibilities

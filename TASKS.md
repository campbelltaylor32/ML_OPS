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
4. ✅ AutoML pipeline (FLAML + Baseline + Top-5, MLflow-tracked, DVC-managed)
5. ✅ Model deployed (FastAPI + Streamlit + Docker Compose)
6. ✅ Monitoring + dashboard (PSI + Evidently Cloud + Streamlit Monitoring tab)
7. ✅ Original test data validated against deployed model
8. ✅ Modified test set script (`make_modified_test.py`, 4 features changed)
9. ✅ Modified data run against model; delta metrics + PSI drift verified in monitoring
10. ✅ Presentation structure complete including team member responsibilities

---

## Goal C — Docker-First Consolidation & CI/CD — 2026-05-28

**Audit complete.** Plan approved. Implementing below.

### Consolidation (single DAG source of truth)
- [x] Audit repo for overlapping pipeline runners, dead scripts, stale docs — 2026-05-28
- [x] Add `eda` stage to `dvc.yaml` so `dvc repro` regenerates `reports/figures/` — 2026-05-28
- [x] Change `docker-compose.yml` `pipeline` service command: `bash run_all.sh` → `dvc repro`; drop `mem_limit: 6g` / H2O comment — 2026-05-28
- [x] Update `run_all.sh` header to note `dvc repro` is canonical — 2026-05-28
- [x] Delete `scripts/ci_pipeline.sh` (replaced by `dvc repro` in CI) — 2026-05-28

### Dead-code & stale-config removal
- [x] Delete `src/train_automl.py` (unreferenced duplicate of FLAML path); inline `build_preprocessor` into `automl_benchmark.py` — 2026-05-28
- [x] Delete `pytest.ini`; add `pythonpath = ["."]` to `[tool.pytest.ini_options]` in `pyproject.toml` — 2026-05-28
- [x] Strip all H2O references from `README.md` (team table, Pillar 3 table, results table, folder structure, `uv sync --extra h2o`) — 2026-05-28
- [x] Remove `train_automl` command line from `CLAUDE.md` — 2026-05-28

### Docker verify surface
- [x] Add `test` profile service to `docker-compose.yml` (`uv run python -m pytest -q` inside image) — 2026-05-28
- [x] Add `scripts/verify_docker.sh` — canonical one-liner verify loop (pipeline → test → smoke) — 2026-05-28

### GitHub Actions (complete MLOps CI/CD)
- [x] Rewrite `ci.yml`: two jobs — `lint` (ruff) + `build-test` (Docker-first: `dvc repro` + pytest + smoke inside image) — 2026-05-28
- [x] Add `publish.yml`: build + push to `ghcr.io/campbelltaylor32/ml_ops` on `main` push / tags — 2026-05-28
- [x] Add `drift.yml`: weekly cron + `workflow_dispatch` — runs monitoring inside image, uploads report artifact — 2026-05-28
- [x] Add `ruff` to `pyproject.toml` dev deps + `[tool.ruff]` config; regenerate `uv.lock` — 2026-05-28

### Docs
- [x] Update `README.md`: Docker-first run section, GHCR pull instructions, remove H2O, fix folder tree — 2026-05-28
- [x] Update `CLAUDE.md`: remove `train_automl`, note `dvc repro` canonical, `ci_pipeline.sh` gone, stage count → 10 — 2026-05-28

---

## Goal D — CI/CD Trigger Strategy — handoff

**Problem:** `publish.yml` currently triggers on both `branches: [main]` and `tags: ["v*.*.*"]`, causing two concurrent Docker builds on every push to `main` (CI + Publish run in parallel, wasting build minutes and potentially publishing unverified images).

**Task:** Investigate and plan the right publish trigger strategy before implementing.

### Research questions
- [x] Should publish run tags-only (`v*.*.*`) — standard release cadence — or should `main` always publish `latest`?
- [x] If tags-only: what is the right tagging workflow for a course/team project (lightweight vs annotated tags, who tags, when)?
- [x] If both: should `publish` depend on `ci` passing first (e.g. via `workflow_run` event or merging into one workflow)?
- [x] Does `drift.yml` need to pull the GHCR image or can it build fresh — and does that affect the trigger decision?

### Decision: Option 3 — Single Workflow
- [x] Delete `publish.yml` — replaced by `publish` job in `ci.yml`
- [x] Add `tags: ["v*.*.*"]` to `ci.yml` `on.push` trigger
- [x] Add `publish` job to `ci.yml` gated by `needs: build-test`, `if: github.event_name == 'push'`
- [x] `drift.yml` unchanged — pulls `:latest` which is still published after CI passes on main — 2026-05-28

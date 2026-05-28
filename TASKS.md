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

- [ ] Add `dvc metrics` tracking for accuracy, F1, PSI scores
- [ ] Integrate DVC pipeline status badge into README

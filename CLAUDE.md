# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CodeGraph — Default First Step (Every Prompt)

**Before any file read, grep, or edit**, orient with CodeGraph MCP tools. This project has a live index (`codegraph_status` to verify). The workflow on every prompt:

1. **Understand the area** → `codegraph_context` on the relevant symbol/task name
2. **Find a symbol** → `codegraph_search` (never grep-first for symbol lookups)
3. **Trace a flow** → `codegraph_trace` from→to (one call, full path)
4. **Survey several symbols** → `codegraph_explore` (one capped call, not multiple `Read`s)
5. **Check blast radius before editing** → `codegraph_impact`

Never spawn an Explore agent or run a grep/Read loop when 1–2 codegraph calls answer the question. Raw `Read`/`grep` is reserved for: literal string content, comments, log messages, or a specific line after codegraph surfaced the file.

**Hard rules — violations recorded:**
- `codegraph_context` is the **sole first call** on every prompt. Never call `codegraph_files` or any other codegraph tool in parallel with it for initial orientation — `codegraph_context` already composes search + node + callers + callees.
- Never pre-announce "then I'll check the config / Docker files with Read" before seeing codegraph results. Wait for codegraph output; only reach for `Read`/`grep` if codegraph explicitly didn't cover that file.
- codegraph tools are deferred — `ToolSearch` must load them first. Do NOT delay orientation: load and call `codegraph_context` in the same turn, not across two turns.

## Commands

```bash
# Environment setup (uv — installs all deps from uv.lock)
uv sync                           # create/update .venv

# ── Docker (canonical path) ──────────────────────────────────────────────────
docker compose run --rm pipeline    # run dvc repro inside the image; artifacts land on host
docker compose run --rm test        # run pytest inside the image against pipeline artifacts
docker compose up                   # start mlflow (5001) + api (8000) + dashboard (8501)
bash scripts/verify_docker.sh       # all three steps above in one command

# ── Local fallback (no Docker) ───────────────────────────────────────────────
bash run_all.sh                     # mirrors dvc.yaml stage order; use when DVC unavailable

# Individual pillars (local, for dev iteration)
uv run python -m src.eda                               # EDA charts -> reports/figures/
uv run python -m src.lineage.run_lineage               # Lineage: Raw→Staging→Intermediate→Marts + diagram
uv run python -m src.preprocessing                     # CSV-compat train/test split -> data/processed/
uv run python -m src.feature_store                     # Feature Store: materialise v1 + v2
uv run python -m src.experiment                        # Experiment grid: feat×hparams + CodeCarbon
uv run python -m src.automl_benchmark --flaml-budget 60 # AutoML: Baseline/FLAML + latency + top-5
uv run python -m src.evaluate                          # Metrics on original test set
uv run python -m src.make_modified_test                # Build "academic stress" test set
uv run python -m src.batch_inference                   # Predict both sets + compare
uv run python -m src.monitoring                        # PSI drift report
uv run python -m src.monitoring_evidently              # Evidently DataDrift+DataSummary (+ Cloud if token set)

# Tests — use uv run (resolves .venv correctly; bare pytest will ImportError on src)
uv run python -m pytest -q

# Lint
uv run ruff check .
uv run ruff format --check .

# Serving — local fallback
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # http://localhost:5001
uv run streamlit run app/dashboard.py                       # http://localhost:8501
uv run uvicorn app.inference_api:app --port 8000            # http://localhost:8000/docs
bash serve_demo.sh                                          # all three at once
```

## Architecture

**Single config source of truth:** `src/config.py` defines all paths (including lineage, feature store, and report dirs), feature lists, random seed, MLflow URIs, and env var keys for Evidently Cloud. Every script imports from it.

**Four-pillar enhanced pipeline flow:**
```
raw CSV
  └─[lineage/run_lineage.py]──▶ data/staging/ → data/intermediate/ → data/marts/
       └─[feature_store.py]──▶ data/feature_store/{v1,v2}/
            └─[experiment.py]──▶ MLflow runs (feat×hparams, CodeCarbon emissions)
                 └─[automl_benchmark.py]──▶ Baseline/FLAML + latency + top-5 → models/best_model.pkl
                      └─[monitoring.py + monitoring_evidently.py]──▶ PSI + Evidently reports
```

**Lineage contracts (checking guards):** `src/lineage/contracts.py` defines `validate_staging()`,
`validate_intermediate()`, `validate_marts()`, `validate_feature_store()`, `validate_model_ready()`.
Each raises on violation. Called at the top of every downstream module's main.

**Feature store:** `src/feature_store.py` has `FeatureStore.get_feature_set(version)` which returns
`(X_train, X_test, y_train, y_test, meta)`. Version `v2` adds `study_sleep_ratio` + `effective_attendance`.

**MLflow wrappers:** `src/mlflow_utils.py` has `start_tracked_run`, `log_feature_set`, `track_emissions`.
Always use these instead of raw `mlflow.start_run` so carbon tracking is consistent.

**Model artifact:** `automl_benchmark.py` saves the best FLAML pipeline to `models/best_model.pkl`
and registers it. `app/` feeds raw rows — preprocessing is baked into the pipeline.

**Monitoring:** Two tracks: `src/monitoring.py` (PSI, dependency-free) and `src/monitoring_evidently.py`
(Evidently DataDrift + DataSummary; Cloud-upload when `EVIDENTLY_API_TOKEN` env var is set).

## Key constraints

- **No leakage columns:** `Math_Score`, `Science_Score`, `Language_Score`, `History_Score`, `Grade` must never reach the model. Validated by `contracts.validate_marts()`.
- **MLflow tracking URI must be SQLite** (`sqlite:///mlflow.db`) — required for the Model Registry locally.
- **`models/best_model.pkl` is the serving artifact** — apps load this directly; do not depend on MLflow being live.
- **`data/processed/`, `data/staging/`, `data/intermediate/`, `data/marts/`, `data/feature_store/`, `reports/` are all generated** — do not commit them. Run `docker compose run --rm pipeline` (or `bash run_all.sh` locally) to regenerate.
- **Dependency pinning — two-file strategy:** `requirements.txt` holds pinned direct deps (`==`); `requirements.lock` is the full `pip freeze` for exact transitive reproduction. Always install from `requirements.lock` for a reproducible env. After any version bump: update `requirements.txt`, run `bash run_all.sh`, then `pip freeze > requirements.lock`. Never use `>=` ranges in `requirements.txt` — the evidently 0.6/0.7 API breakage in `monitoring_evidently.py` is the canonical example of why.
- **Evidently Cloud credentials live in `.env`** — copy `.env.example` → `.env` and fill in `EVIDENTLY_API_TOKEN` + `EVIDENTLY_PROJECT_ID`. `src/config.py` auto-loads it via `python-dotenv`. `.env` is gitignored; never commit it. Local HTML is always written regardless of token.

## Data & Versioning (DVC)

**DVC is initialized** (`dvc init` already run). The pipeline is defined in `dvc.yaml` with 10 stages: `eda → lineage → preprocessing → feature_store → experiment → automl_benchmark → evaluate → make_modified_test → batch_inference → monitoring → monitoring_evidently`. `dvc repro` is the canonical pipeline runner; `docker compose run --rm pipeline` executes it inside the image.

**DVC commands:**
```bash
.venv/bin/dvc repro              # Reproduce full pipeline (skips unchanged stages)
.venv/bin/dvc status             # Show which stages are stale
.venv/bin/dvc params diff        # Diff params.yaml changes vs last run
.venv/bin/dvc dag                # Print the pipeline DAG
.venv/bin/dvc add data/raw/student_performance.csv  # Track raw data with DVC
```

**Integration rule — AutoML ↔ MLflow ↔ DVC:**
- Every `automl_benchmark` run must log the DVC data hash to MLflow: call `mlflow.log_param("dvc_data_hash", _get_dvc_hash("data/feature_store"))` inside `src/automl_benchmark.py` via `src/mlflow_utils.py`.
- Every update to `data/raw/student_performance.csv` (i.e., `dvc add data/raw/…`) must be followed by `dvc repro` to regenerate lineage → features → model → Evidently drift report. The `monitoring_evidently` stage is a downstream dep of `automl_benchmark` in `dvc.yaml`, so `dvc repro` handles the cascade automatically.

**DVC-tracked paths (do not manually edit):**
- `data/raw/student_performance.csv.dvc` — raw data pointer (commit this, not the CSV if large)
- `dvc.lock` — locked hashes after a successful `dvc repro` (commit alongside code changes)
- `params.yaml` — pipeline hyperparams; `automl.flaml_budget` controls FLAML time budget

**What DVC does NOT track (handled by MLflow or gitignored):**
- `mlflow.db`, `mlruns/` — MLflow owns these

## Task & Planning Protocol

**Before modifying any core code**, create `IMPLEMENTATION_PLAN.md` in the project root with:
1. Task title and motivation
2. Files to be changed (with line-number anchors where relevant)
3. Numbered implementation steps
4. Test/validation plan
5. Rollback path

Wait for user approval of `IMPLEMENTATION_PLAN.md` before making any edits to `src/`, `app/`, `dvc.yaml`, or `requirements.txt`.

**Task tracking:** `TASKS.md` is the persistent task tracker. Update it at the start and end of every work session. Format: `[ ]` open, `[x]` done, `[~]` in-progress.

**Per-session checklist:**
1. Read `MEMORY.md` index + relevant memory files
2. Read `TASKS.md` and mark in-progress items `[~]`
3. For new tasks: write `IMPLEMENTATION_PLAN.md`, get approval, then implement
4. After each logical unit: run `uv run python -m pytest -q` and `uv run dvc status`
5. End of session: update `TASKS.md`, write new memory files, delete `IMPLEMENTATION_PLAN.md`

## Anti-patterns and rules (from lessons_learned.md)

- **Never call `pytest` bare** — always use `uv run python -m pytest -q`. The system pytest resolves a different `src` and will always fail with `ImportError: cannot import name 'config' from 'src'`.
- **Verify third-party constructor signatures before use** — run `python -c "import inspect; from pkg import Cls; print(inspect.signature(Cls.__init__))"`. Known traps:
  - `evidently.Report`: parameter is `metrics=`, NOT `items=` (changed in 0.7.x)
  - `evidently.ui.workspace.CloudWorkspace`: import from `evidently.ui.workspace`, NOT `.workspace.cloud`
- **Every new `import <pkg>` must appear in both `pyproject.toml` and `requirements.txt`** in the same change, then regenerate with `uv lock && uv sync`.
- **Optional heavy deps (Evidently Cloud, CodeCarbon) must be wrapped** in `try/except ImportError` and skip gracefully — never let them crash the core pipeline.
- **`ColumnTransformer` drops feature names** — if downstream code needs named features (importance, SHAP), add `.set_output(transform="pandas")` to the transformer (sklearn ≥ 1.2).
- **`src/lineage/contracts.py` guards must be called** at the top of any module that reads a downstream artifact — do not assume upstream validity.
- **CodeCarbon log level must be set after import, not before** — codecarbon's `external/logger.py` resets the logger to INFO on import, overriding any pre-import suppression. Set `logging.getLogger("codecarbon").setLevel(logging.ERROR)` inside `track_emissions`, after the `from codecarbon import ...` line (lessons_learned.md #6).

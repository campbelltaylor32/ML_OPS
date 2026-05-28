# Lessons Learned — Student Performance MLOps Pipeline

Errors encountered during implementation, root causes, fixes, and prevention rules.

---

## Error 1 — Evidently `Report()` constructor keyword argument changed

**Error:**
```
TypeError: Report.__init__() got an unexpected keyword argument 'items'
```
**File:** `src/monitoring_evidently.py`
**When:** Called `Report(items=[DataDriftPreset(), DataSummaryPreset()])` on evidently 0.7.x

**Root cause:**
The Evidently API changed the constructor parameter name from `items` (used in many
online tutorials and the ≤ 0.4.x docs) to `metrics` in 0.7.x. Both names existed
in older versions; only `metrics` is valid in 0.7+.

**Fix:**
```python
# Wrong (≤ 0.4.x style — still shown in most tutorials)
report = Report(items=[DataDriftPreset(), DataSummaryPreset()])

# Correct (0.7.x)
report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
```

**Prevention rule:**
Before writing code that calls a third-party constructor, verify the live signature:
```python
import inspect
from evidently import Report
print(inspect.signature(Report.__init__))
```
Do this for every new library API surface before implementation, not after. This is
especially critical for Evidently, MLflow, and CodeCarbon which all evolve quickly.

---

## Error 2 — sklearn `UserWarning: X does not have valid feature names`

**Error (warning, not crash):**
```
UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
```
**File:** `src/experiment.py`
**When:** `Pipeline.predict()` passes a numpy array (from `ColumnTransformer` output)
into an estimator that was fitted on a named DataFrame.

**Root cause:**
`ColumnTransformer.fit_transform()` returns a numpy array, discarding column names.
When that array is passed to `LGBMRegressor`, the estimator sees unnamed input even
though it was fitted with named features. This is a sklearn / LightGBM mismatch in
name tracking through the pipeline — harmless to predictions but noisy.

**Fix options:**
1. Wrap the `ColumnTransformer` output in `pd.DataFrame` with `get_feature_names_out()`
   before fitting the estimator (adds overhead, rarely worth it for regression).
2. Set `LGBMRegressor(feature_name='auto')` — tells LightGBM to accept unnamed input silently.
3. Silence at the warning level (last resort, hides real issues).

**Chosen mitigation:** Accept the warning — it is not a correctness issue for the
regression pipeline. The warning is suppressed in `pytest` output via `filterwarnings`.
If LightGBM feature importance is required by name (e.g. in `automl_benchmark.py`
for top-5 selection), use `get_feature_names_out()` before fitting there specifically.

**Prevention rule:**
When chaining `ColumnTransformer → estimator` in a `Pipeline`, the transformer
drops feature names. If the downstream estimator needs named features (for
importance, SHAP, or logging), wrap the transformer output with
`set_output(transform="pandas")` on the `ColumnTransformer`:
```python
pre = ColumnTransformer(...).set_output(transform="pandas")
```
This is available in sklearn ≥ 1.2 and propagates names through the pipeline.

---

## Error 3 — `evidently.ui.workspace.cloud` import path wrong

**Error:**
```
ImportError: No module named 'evidently.ui.workspace.cloud'; 'evidently.ui.workspace' is not a package
```
**File:** `src/monitoring_evidently.py`
**When:** Tried `from evidently.ui.workspace.cloud import CloudWorkspace`

**Root cause:**
In evidently 0.7.x, `CloudWorkspace` lives at `evidently.ui.workspace`, not in a
`.cloud` sub-module. The `.cloud` sub-path exists in evidently ≥ 0.5 documentation
examples but was consolidated in 0.7.

**Fix:**
```python
from evidently.ui.workspace import CloudWorkspace
```

**Prevention rule:**
Never trust the import path from online examples or official docs without verifying
against the installed version:
```python
python -c "from evidently.ui.workspace import CloudWorkspace; print('OK')"
```
Run this check for all Cloud/remote API imports before writing the module.

---

## Error 4 — `pytest` using system Python, not `.venv`

**Error:**
```
ImportError: cannot import name 'config' from 'src' (unknown location)
```
**When:** Running `pytest -q` instead of `.venv/bin/python -m pytest -q`

**Root cause:**
The shell's `pytest` resolved to the system-installed pytest (via `miniforge3`),
which sees a different `src` package (likely empty or stale from a different
Python path). The `.venv` pytest correctly resolves to the project's `src/`.

**Fix:**
Always invoke pytest through the project venv:
```bash
.venv/bin/python -m pytest -q
# or after activating the venv:
source .venv/bin/activate && pytest -q
```

**Prevention rule:**
Add a `pytest.ini` or `pyproject.toml` at the repo root with `pythonpath = .` so
pytest adds the project root to `sys.path` regardless of how it is invoked:
```ini
# pytest.ini
[pytest]
pythonpath = .
```
This makes `pytest -q` (any interpreter) and `.venv/bin/python -m pytest -q`
behave identically, preventing environment confusion.

---

## Error 5 — `codecarbon` not in `requirements.txt`

**Symptom:** `pip show codecarbon` returned `WARNING: Package(s) not found`.

**Root cause:**
`codecarbon` was listed in the implementation plan as a new dependency but was
missing from `requirements.txt`. It was present in the venv only after an ad-hoc
`pip install codecarbon` during the session.

**Fix:** Added `codecarbon>=2.0` to `requirements.txt`.

**Prevention rule:**
Every `import <new_package>` introduced in a new module must have a corresponding
entry in `requirements.txt` in the same commit. Check before closing the PR:
```bash
grep -rh "^import\|^from" src/ | awk '{print $2}' | cut -d. -f1 | sort -u
```
Cross-reference the output against `requirements.txt`.

---

---

## Error 6 — CodeCarbon "Multiple instances" warning not silenced by `log_level` parameter

**Error (warning):**
```
[codecarbon WARNING @ HH:MM:SS] Multiple instances of codecarbon are allowed to run at the same time.
```

**Root cause:**
`EmissionsTracker.__init__` emits the warning at **line 310** of `emissions_tracker.py`.
The `log_level` parameter (passed to the constructor) is applied at **line 390** via
`set_logger_level()` — *after* the warning already fired.  Passing `log_level="error"`
therefore has no effect on this specific warning.

Additionally, `codecarbon/external/logger.py` runs `set_logger_level()` with the
default "INFO" during module import, which resets any suppression set *before* the
import.

**Fix:**
Set `logging.getLogger("codecarbon").setLevel(logging.ERROR)` **inside the context
manager, after the `from codecarbon import EmissionsTracker` line, and before
`EmissionsTracker(...)` is instantiated**:
```python
from codecarbon import EmissionsTracker
import logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)  # override post-import reset
tracker = EmissionsTracker(...)
```

**Prevention rule:**
For any library that configures Python's `logging` during its own import (CodeCarbon,
Evidently, H2O), always set `logging.getLogger("<pkg>").setLevel(...)` AFTER the
import statement, not before. Setting it before is silently overridden.

---

## Error 7 — H2O `aml.leader.model_id` accessed after `h2o.cluster().shutdown()`

**Error:**
```
h2o.exceptions.H2OConnectionError: Connection was closed, and can no longer be used.
  File "src/automl_benchmark.py", line N, in run_h2o
      return {"regime": "h2o", "leader": aml.leader.model_id, ...}
```

**Root cause:**
In `run_h2o`, `h2o.cluster().shutdown()` was called on the line immediately before
the `return` statement. `aml.leader` is a lazy H2O property — accessing `.model_id`
makes an HTTP request to the H2O server. After `shutdown()`, the server is gone and
the request fails with a connection error.

The same `aml.leader.model_id` was accessed earlier (inside the `with` block for
`mlflow.log_params`) where it worked correctly because the server was still running.

**Fix:**
Capture any H2O model attributes that will be used in the return value **before**
calling `h2o.cluster().shutdown()`:
```python
leader_id = aml.leader.model_id  # capture while connection is live
h2o.cluster().shutdown()
return {"regime": "h2o", "leader": leader_id, ...}
```

**Prevention rule:**
Any attribute access on an H2O object (`H2OModel`, `H2OFrame`, `H2OAutoML.leader`)
that requires a network round-trip to the JVM server must happen **before**
`h2o.cluster().shutdown()`. After shutdown, all H2O Python proxies are dead handles.
Prefer capturing all values inside the `with` block or before the shutdown line.

---

## Error 8 — Evidently Cloud token stored in shell env / settings files instead of `.env`

**Symptom:** Token had to be exported manually each session or stored in `.claude/settings.local.json`,
which is not portable across machines and not visible to the Python process.

**Root cause:**
`src/config.py` read `EVIDENTLY_API_TOKEN` only from `os.environ`, so the token had to
be injected via `export` or a Claude Code env block. Neither approach is reproducible for
teammates or CI.

**Fix:**
1. Add `python-dotenv>=1.0` to `requirements.txt`.
2. Create `.env.example` (committed) with placeholder values and instructions.
3. Each developer copies `.env.example` → `.env` (gitignored) and fills in their token.
4. In `src/config.py`, call `load_dotenv(ROOT / ".env")` before the `os.environ.get` calls:
   ```python
   from dotenv import load_dotenv as _load_dotenv
   _load_dotenv(ROOT / ".env")
   EVIDENTLY_API_TOKEN: str = _os.environ.get("EVIDENTLY_API_TOKEN", "")
   ```
   `load_dotenv` is a no-op when `.env` is absent, so CI (which injects vars via the runner)
   works unchanged.

**Prevention rule:**
Any secret that a script reads from `os.environ` should have a corresponding `.env.example`
entry and a `load_dotenv()` call at the config layer. Never store real tokens in committed
files or Claude Code settings — use `.env` (gitignored) as the canonical local secret store.

---

## Error 9 — Loose `>=` version ranges caused silent API breakage on merge

**Symptom:** After merging remote changes, `monitoring_evidently.py` crashed with
`ModuleNotFoundError: No module named 'evidently.report'` and
`AttributeError: 'Report' object has no attribute 'save_html'`.

**Root cause:**
`requirements.txt` used `evidently>=0.7`, which allowed any 0.7.x install.
The merged code was written against 0.6.7 import paths (`evidently.report.Report`,
`evidently.metric_preset.DataQualityPreset`). The local env had `evidently==0.7.21`
where both import paths and method names had changed. Loose ranges make it impossible
to know which API surface a contributor was writing against.

**Fix:**
1. Pin all direct dependencies to exact versions (`==`) in `requirements.txt`.
2. Commit a full `pip freeze` snapshot as `requirements.lock` for transitive reproducibility.
3. Install via `pip install -r requirements.lock` for exact environment reproduction.

**Prevention rule:**
Use `>=` only in library packages (which must stay compatible with their dependents).
In application projects, always use `==` in `requirements.txt`. The upgrade workflow is:
bump version in `requirements.txt` → install → run `bash run_all.sh` → `pip freeze > requirements.lock`.

---

## Error 10 — H2O `verbosity=None` is accepted but suppresses leaderboard output

**Symptom:**
No per-model progress output from H2O AutoML; the section runs silently beyond the
`h2o.no_progress()` call. Not an error — this is the desired behavior.

**Root cause:**
`H2OAutoML(verbosity=None)` suppresses the leaderboard print at the end of training.
The signature default is `verbosity='warn'`; passing `None` is valid and means "silent".

**Prevention rule:**
Keep `verbosity=None` when running in pipeline mode (avoids noise). Use `verbosity='warn'`
or `verbosity='info'` when debugging an H2O training run interactively.

---

## Error 11 — `run_flaml` registered every run to MLflow Model Registry unconditionally

**Symptom:**
Model Registry accumulated new versions on every `automl_benchmark` run regardless of
whether FLAML was actually the best model across all regimes.

**Root cause:**
`mlflow.sklearn.log_model(registered_model_name=...)` inside `run_flaml()` always
registered FLAML's pipeline, bypassing the multi-regime comparison that happens later
in `main()`. This meant H2O winning didn't affect which artifact was promoted.

**Fix:**
1. Removed `registered_model_name` from `mlflow.sklearn.log_model` inside `run_flaml()` —
   the model artifact is still logged to the run, just not registered.
2. Added a dedicated `model_selection` MLflow run in `main()` that registers only the
   winner after all regimes are compared:
   ```python
   mlflow.sklearn.log_model(
       sk_model=best_sklearn["pipeline"],
       name="model",
       registered_model_name=config.REGISTERED_MODEL_NAME,
       input_example=X_train.head(2),
   )
   ```
3. Model metadata now includes `deployed_regime`, `best_overall_regime`, and
   `all_regimes` so the comparison result is always auditable.

**Prevention rule:**
Never register a model to the MLflow Model Registry inside an individual regime function.
Registration is a promotion decision — always make it in the orchestrating `main()` after
all candidates have been evaluated.

---

## Error 12 — H2O model cannot be serialized as `best_model.pkl` (sklearn Pipeline)

**Symptom:**
H2O `aml.leader` is a JVM proxy object. `joblib.dump(aml.leader, ...)` would fail or
produce an unloadable file because the H2O Python client stores references to a live JVM
process, not in-process Python objects.

**Root cause:**
The serving layer (`app/inference_api.py`, `app/dashboard.py`) loads `models/best_model.pkl`
and calls `.predict()` on it as a sklearn-compatible object. H2O models are incompatible
with this interface.

**Fix:**
In `main()`, `sklearn_candidates` is restricted to `[baseline_res, flaml_res]` (regimes
that carry a `"pipeline"` key). H2O participates in the RMSE comparison (`all_scored`)
to determine `best_overall_regime`, but the **deployed** model is always the best sklearn
pipeline. Metadata records both:
```json
{
  "deployed_regime": "flaml",
  "best_overall_regime": "h2o"
}
```
This makes it visible when H2O wins so the team can decide to invest in H2O MOJO export
or an h2o-wave serving layer as a follow-up.

**Prevention rule:**
When adding a new AutoML regime, check whether it produces a `joblib`-serializable sklearn
`Pipeline` before adding it to `sklearn_candidates`. H2O, AutoGluon, and other JVM/native
backends must stay outside that list until a sklearn-compatible wrapper exists.

---

## General rules derived from this session

1. **Verify all third-party API signatures before writing code** — one
   `inspect.signature(Cls.__init__)` call costs nothing; a `TypeError` at runtime
   costs a full debug cycle.

2. **Always use `.venv/bin/python -m pytest`** — never the bare `pytest` command
   unless the venv is activated and confirmed active.

3. **Guard optional heavy dependencies at import time** — H2O, Evidently Cloud, and
   CodeCarbon are all optional. Wrap with `try/except ImportError` and log a clear
   skip message; never let a missing dep crash the core pipeline.

4. **One `inspect.signature` call per new library surface** — add this to the
   "Hallucination Guard" step: before writing a call to any external function, print
   its signature and verify every keyword argument exists.

5. **Lineage contracts run before feature materialisation** — never assume upstream
   parquet is valid; always call `validate_*()` as the first line of any downstream
   module's `main()`.

6. **MLflow model registration is a promotion decision, not a per-run side-effect** —
   never call `mlflow.sklearn.log_model(registered_model_name=...)` inside individual
   regime functions. Register only in the orchestrating `main()` after cross-regime
   comparison.

7. **H2O and other JVM-backed AutoML frameworks cannot be directly serialized as
   sklearn Pipelines** — always restrict `sklearn_candidates` to regimes that return
   a `"pipeline"` key; include all regimes in `all_scored` for RMSE comparison only.
   Record `deployed_regime` vs `best_overall_regime` in metadata so the gap is visible.

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

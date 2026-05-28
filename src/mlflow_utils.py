"""
mlflow_utils.py  --  Thin MLflow + CodeCarbon wrappers
=======================================================
Centralises boilerplate so every tracking site stays consistent.
All wrappers delegate to the real MLflow/CodeCarbon APIs; nothing is cached.
"""

import contextlib
import json
import logging
import tempfile
from typing import Any, Dict, Generator, Optional

import mlflow
import mlflow.sklearn

from src import config


def _get_dvc_hash(path: str) -> str:
    lock = config.ROOT / "dvc.lock"
    if not lock.exists():
        return "unknown"
    import yaml

    data = yaml.safe_load(lock.read_text()) or {}
    for stage in (data.get("stages") or {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == path:
                return out.get("md5") or out.get("hash") or "unknown"
    return "unknown"


def setup_mlflow(experiment: str = config.MLFLOW_EXPERIMENT) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)


@contextlib.contextmanager
def start_tracked_run(
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    experiment: str = config.MLFLOW_EXPERIMENT,
) -> Generator[mlflow.ActiveRun, None, None]:
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_feature_set(meta: Dict[str, Any]) -> None:
    """Log feature-set version metadata to the active MLflow run."""
    mlflow.set_tag("feature_version", meta.get("version", "unknown"))
    mlflow.log_param("feature_count", meta.get("n_features", 0))
    mlflow.log_param("feature_version", meta.get("version", "unknown"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(meta, f, indent=2)
        tmp = f.name
    mlflow.log_artifact(tmp, artifact_path="feature_meta")


@contextlib.contextmanager
def track_emissions(project_name: str = "student_performance") -> Generator[None, None, None]:
    """
    Context manager: wraps EmissionsTracker and logs co2_kg_emissions to the
    active MLflow run when the block exits.  Falls back silently if codecarbon
    is unavailable or the active run is None.
    """
    try:
        from codecarbon import EmissionsTracker

        # codecarbon's external/logger.py resets the logger to INFO on import;
        # override it here, after the import, so the "Multiple instances" warning
        # (fired at __init__ line 310, before set_logger_level at line 390) is suppressed.
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        tracker = EmissionsTracker(
            project_name=project_name,
            save_to_file=False,
            save_to_api=False,
            log_level="error",
            allow_multiple_runs=True,
        )
        tracker.start()
        try:
            yield
        finally:
            emissions_kg = tracker.stop()
            if emissions_kg is not None and mlflow.active_run() is not None:
                mlflow.log_metric("co2_kg_emissions", round(float(emissions_kg), 8))
                mlflow.set_tag("carbon_tracked", "true")
    except ImportError:
        yield
    except Exception:
        yield

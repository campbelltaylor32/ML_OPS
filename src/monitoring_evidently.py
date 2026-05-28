"""
monitoring_evidently.py  --  Evidently drift + summary report (local + Cloud)
==============================================================================
Uses DataDriftPreset + DataSummaryPreset (evidently >= 0.4).
Cloud upload is token-gated: set EVIDENTLY_API_TOKEN env var to enable.
Offline / no-token path always works and writes a local HTML report.

Run:  python -m src.monitoring_evidently
"""
import os
from pathlib import Path

import pandas as pd

from src import config


def _build_report(reference: pd.DataFrame, current: pd.DataFrame):
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    return report.run(reference_data=reference, current_data=current)


def _save_local(snapshot, out: Path) -> Path:
    snapshot.save_html(str(out))
    return out


def _upload_cloud(snapshot) -> bool:
    token = config.EVIDENTLY_API_TOKEN
    project_id = config.EVIDENTLY_PROJECT_ID
    url = config.EVIDENTLY_CLOUD_URL

    if not token:
        return False

    try:
        from evidently.ui.workspace import CloudWorkspace
        ws = CloudWorkspace(token=token, url=url)
        ws.verify()

        if project_id:
            ws.add_run(project_id, snapshot)
            print(f"Evidently Cloud: uploaded to project {project_id}")
        else:
            projects = ws.list_projects()
            if projects:
                ws.add_run(projects[0].id, snapshot)
                print(f"Evidently Cloud: uploaded to project {projects[0].name}")
            else:
                print("Evidently Cloud: no project found; set EVIDENTLY_PROJECT_ID")
                return False
        return True
    except Exception as exc:
        print(f"Evidently Cloud upload failed (continuing offline): {exc}")
        return False


def run() -> Path:
    config.ensure_dirs()

    reference = pd.read_csv(config.TEST_PATH)[config.FEATURES]
    current = pd.read_csv(config.TEST_MODIFIED_PATH)[config.FEATURES]

    snapshot = _build_report(reference, current)

    out = config.MONITORING_DIR / "evidently_report.html"
    _save_local(snapshot, out)
    print(f"Evidently report (local) -> {out.relative_to(config.ROOT)}")

    if config.EVIDENTLY_API_TOKEN:
        uploaded = _upload_cloud(snapshot)
        if not uploaded:
            print("Cloud upload not completed; local report is the primary output.")
    else:
        print("EVIDENTLY_API_TOKEN not set — local mode only.")

    return out


if __name__ == "__main__":
    run()

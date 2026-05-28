"""
monitoring_evidently.py  --  OPTIONAL Evidently AI drift report
================================================================
MLOps purpose: an industry-standard MONITORING report as an alternative to the
built-in PSI monitor in src/monitoring.py. Produces an HTML report comparing the
original (reference) and modified (current) test sets.

This is OPTIONAL and kept separate so an Evidently version mismatch can never
break the core demo. If Evidently is not installed, the script prints how to
install it and exits cleanly.

Run:  python -m src.monitoring_evidently
Open: reports/monitoring/evidently_report.html
"""
import pandas as pd

from src import config


def run():
    config.ensure_dirs()
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:
        # Older Evidently API fallback
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except Exception:
            print("Evidently not installed or API changed. Install with:")
            print("    pip install evidently")
            print("The built-in monitor (src/monitoring.py) works without it.")
            return None

    reference = pd.read_csv(config.TEST_PATH)[config.FEATURES]
    current = pd.read_csv(config.TEST_MODIFIED_PATH)[config.FEATURES]

    out = config.MONITORING_DIR / "evidently_report.html"
    try:
        # Newer API (evidently >= 0.4.x "Report.run" then save_html)
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        result.save_html(str(out))
    except Exception:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(out))

    print(f"Evidently report saved -> {out.relative_to(config.ROOT)}")
    return out


if __name__ == "__main__":
    run()

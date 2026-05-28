"""
make_modified_test.py  --  Step 9: create the changed test dataset
==================================================================
MLOps purpose: SIMULATE INPUT DATA DRIFT to test monitoring.
We apply an "academic stress" scenario to the test set, changing 4 behavioural
features (>=2 required by the assignment):
    Study_Hours_Per_Day  x0.5      (students study half as much)
    Stress_Level         +3        (stress spikes)
    Sleep_Hours_Per_Day  -1.5      (less sleep)
    Attendance_Rate      -15       (more absences)
Values are clipped to plausible ranges. The TARGET is left unchanged so we can
still measure metric degradation on the same students under stress conditions.

Run:  python -m src.make_modified_test
"""

import pandas as pd

from src import config


def apply_scenario(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (op, val) in config.STRESS_SCENARIO.items():
        if col not in out.columns:
            continue
        if op == "multiply":
            out[col] = out[col] * val
        elif op == "add":
            out[col] = out[col] + val
        if col in config.CLIP_RANGES:
            lo, hi = config.CLIP_RANGES[col]
            out[col] = out[col].clip(lo, hi)
    return out


def make_modified():
    config.ensure_dirs()
    test_df = pd.read_csv(config.TEST_PATH)
    modified = apply_scenario(test_df)
    modified.to_csv(config.TEST_MODIFIED_PATH, index=False)

    print("Created modified ('academic stress') test set.")
    print(f"Changed features: {list(config.STRESS_SCENARIO)}")
    for col in config.STRESS_SCENARIO:
        print(f"  {col:22s} mean {test_df[col].mean():6.2f} -> {modified[col].mean():6.2f}")
    print(f"Saved -> {config.TEST_MODIFIED_PATH.relative_to(config.ROOT)}")
    return modified


if __name__ == "__main__":
    make_modified()

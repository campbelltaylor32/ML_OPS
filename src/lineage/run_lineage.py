"""
run_lineage.py  --  Orchestrate the full Raw→Staging→Intermediate→Mart pipeline
Run:  python -m src.lineage.run_lineage
"""
from src import config
from src.lineage import stg_students, int_features, mart_training
from src.lineage.contracts import validate_staging, validate_intermediate, validate_marts
from src.lineage.lineage_diagram import run as draw_diagram


def main() -> None:
    config.ensure_dirs()

    print("=== Stage 1: Raw → Staging ===")
    stg_students.run()
    validate_staging()
    print("  G0-staging: PASS")

    print("=== Stage 2: Staging → Intermediate ===")
    int_features.run()
    validate_intermediate()
    print("  G0-intermediate: PASS")

    print("=== Stage 3: Intermediate → Marts ===")
    mart_training.run()
    validate_marts()
    print("  G1-marts: PASS — feature store may proceed")

    print("=== Lineage diagram ===")
    draw_diagram()
    print("Lineage pipeline complete.")


if __name__ == "__main__":
    main()

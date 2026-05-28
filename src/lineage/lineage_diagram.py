"""
lineage_diagram.py  --  Emit a Mermaid DAG + render to PNG (if matplotlib available)
"""
import textwrap
from pathlib import Path

from src import config

MERMAID = textwrap.dedent("""\
    graph LR
        A[data/raw/student_performance.csv] --> B[stg_students.parquet]
        B --> C[int_student_features.parquet]
        C --> D[student_training.parquet<br/>student_serving.parquet]
        D --> E1[feature_store/v1]
        D --> E2[feature_store/v2]
        E1 --> F[MLflow Experiment<br/>v1 runs]
        E2 --> F
        F --> G[automl_benchmark.json]
        G --> H[best_model.pkl + MLflow Registry]
        H --> I[monitoring_evidently / PSI drift]
""")


def run() -> Path:
    config.ensure_dirs()
    md_path = config.LINEAGE_DIR / "lineage_dag.md"
    md_path.write_text(f"```mermaid\n{MERMAID}```\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        nodes = [
            ("Raw CSV", 0.05, 0.5),
            ("Staging", 0.22, 0.5),
            ("Intermediate", 0.40, 0.5),
            ("Marts", 0.57, 0.5),
            ("Feature v1/v2", 0.72, 0.5),
            ("MLflow Runs", 0.85, 0.65),
            ("AutoML Bench", 0.85, 0.35),
            ("Model Registry", 0.95, 0.5),
        ]
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        colors = ["#AED6F1", "#A9DFBF", "#FAD7A0", "#F9E79F", "#D7BDE2", "#FADBD8", "#FDEBD0", "#D5F5E3"]
        for (label, x, y), color in zip(nodes, colors):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 0.05, y - 0.12), 0.10, 0.24,
                boxstyle="round,pad=0.01", facecolor=color, edgecolor="gray", linewidth=0.8,
            ))
            ax.text(x, y, label, ha="center", va="center", fontsize=7.5, wrap=True)

        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
        for i, j in edges:
            x0, y0 = nodes[i][1], nodes[i][2]
            x1, y1 = nodes[j][1], nodes[j][2]
            ax.annotate(
                "", xy=(x1 - 0.05, y1), xytext=(x0 + 0.05, y0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            )

        fig.tight_layout()
        png_path = config.LINEAGE_DIR / "lineage_dag.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Lineage PNG -> {png_path.relative_to(config.ROOT)}")
    except Exception as exc:
        print(f"PNG skipped ({exc}); Mermaid .md written.")

    print(f"Lineage Mermaid -> {md_path.relative_to(config.ROOT)}")
    return md_path


if __name__ == "__main__":
    run()

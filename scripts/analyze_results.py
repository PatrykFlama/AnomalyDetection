#!/usr/bin/env python3
"""Create statistics, plots, and a short report from experiment results."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_METRICS = ("point_f1", "point_pr_auc", "event_f1")
DISPLAY_NAMES = {
    "point_f1": "Point F1",
    "point_pr_auc": "Point PR AUC",
    "event_f1": "Event F1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aggregate and per-series anomaly-detection results."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run_inventory(results_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        baseline_summary = run_dir / "baselines" / "summary.csv"
        ts2vec_summary = run_dir / "ts2vec" / "evaluation" / "summary_ts2vec.csv"
        combined_summary = run_dir / "summary.csv"
        rows.append(
            {
                "run_id": run_dir.name,
                "baseline_summary": baseline_summary.is_file(),
                "ts2vec_summary": ts2vec_summary.is_file(),
                "combined_summary": combined_summary.is_file(),
                "complete": baseline_summary.is_file() and ts2vec_summary.is_file(),
            }
        )
    return pd.DataFrame(rows)


def select_run(results_dir: Path, explicit_run_dir: Path | None) -> tuple[Path, pd.DataFrame]:
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    inventory = run_inventory(results_dir)
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.expanduser()
        if not run_dir.is_absolute():
            project_candidate = (PROJECT_ROOT / run_dir).resolve()
            results_candidate = (results_dir / run_dir).resolve()
            run_dir = project_candidate if project_candidate.is_dir() else results_candidate
        else:
            run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory does not exist: {run_dir}")
        return run_dir, inventory

    if inventory.empty:
        raise SystemExit(f"No result runs found in {results_dir}")

    complete = inventory[inventory["complete"]]
    selected_id = (complete if not complete.empty else inventory).iloc[-1]["run_id"]
    return results_dir / str(selected_id), inventory


def load_run_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_paths = [
        run_dir / "baselines" / "summary.csv",
        run_dir / "ts2vec" / "evaluation" / "summary_ts2vec.csv",
    ]
    per_series_paths = [
        run_dir / "baselines" / "per_series_metrics.csv",
        run_dir / "ts2vec" / "evaluation" / "per_series_metrics_ts2vec.csv",
    ]

    summaries = [pd.read_csv(path) for path in summary_paths if path.is_file()]
    per_series_frames = [pd.read_csv(path) for path in per_series_paths if path.is_file()]
    if not summaries:
        raise SystemExit(f"No summary files found under {run_dir}")
    if not per_series_frames:
        raise SystemExit(f"No per-series metric files found under {run_dir}")

    summary = pd.concat(summaries, ignore_index=True, sort=False)
    per_series = pd.concat(per_series_frames, ignore_index=True, sort=False)
    required_summary = {"method", "split", *SUMMARY_METRICS}
    required_per_series = {"method", "split", "series", "point_f1", "event_f1"}
    if missing := required_summary - set(summary.columns):
        raise SystemExit(f"Summary files are missing columns: {sorted(missing)}")
    if missing := required_per_series - set(per_series.columns):
        raise SystemExit(f"Per-series files are missing columns: {sorted(missing)}")
    return summary, per_series


def descriptive_statistics(per_series_test: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "series_count": ("series", "nunique"),
        "point_f1_mean": ("point_f1", "mean"),
        "point_f1_std": ("point_f1", "std"),
        "point_f1_median": ("point_f1", "median"),
        "point_f1_min": ("point_f1", "min"),
        "point_f1_max": ("point_f1", "max"),
        "event_f1_mean": ("event_f1", "mean"),
        "event_f1_std": ("event_f1", "std"),
        "event_f1_median": ("event_f1", "median"),
        "event_f1_min": ("event_f1", "min"),
        "event_f1_max": ("event_f1", "max"),
    }
    if "point_pr_auc" in per_series_test:
        aggregations["point_pr_auc_mean"] = ("point_pr_auc", "mean")
    if "event_fp_events" in per_series_test:
        aggregations["false_positive_events"] = ("event_fp_events", "sum")
    return (
        per_series_test.groupby("method", as_index=False)
        .agg(**aggregations)
        .sort_values("point_f1_mean", ascending=False)
    )


def configure_matplotlib(output_dir: Path):
    cache_dir = Path(tempfile.gettempdir()) / "anomaly-detection-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def save_metric_comparison(plt, test_summary: pd.DataFrame, output_dir: Path) -> None:
    frame = test_summary.sort_values("point_f1", ascending=False).set_index("method")
    axes = frame[list(SUMMARY_METRICS)].plot(
        kind="bar",
        figsize=(12, 6),
        width=0.82,
        color=["#2878B5", "#F8A44C", "#54A24B"],
    )
    axes.set_title("Test-set model comparison")
    axes.set_xlabel("")
    axes.set_ylabel("Score")
    axes.set_ylim(0, 1)
    axes.legend([DISPLAY_NAMES[metric] for metric in SUMMARY_METRICS], loc="upper right")
    axes.tick_params(axis="x", rotation=30)
    axes.figure.tight_layout()
    axes.figure.savefig(output_dir / "test_metric_comparison.png", dpi=180)
    plt.close(axes.figure)


def save_precision_recall(plt, test_summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        ("point_precision", "point_recall", "Pointwise precision and recall"),
        ("event_precision", "event_recall", "Event precision and recall"),
    ]
    for axis, (precision, recall, title) in zip(axes, panels, strict=True):
        axis.scatter(test_summary[recall], test_summary[precision], s=70, color="#2878B5")
        for row in test_summary.itertuples():
            axis.annotate(
                row.method,
                (getattr(row, recall), getattr(row, precision)),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(title)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall.png", dpi=180)
    plt.close(figure)


def save_heatmap(
    plt,
    per_series_test: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    matrix = per_series_test.pivot(index="method", columns="series", values=metric)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    short_names = [Path(name).stem for name in matrix.columns]

    figure_width = max(10, 1.5 * len(short_names))
    figure, axis = plt.subplots(figsize=(figure_width, 0.7 * len(matrix) + 2.5))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(range(len(short_names)), labels=short_names, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set_title(f"Test {DISPLAY_NAMES[metric]} by series")

    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            value = matrix.iloc[row_index, column_index]
            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "black",
                )
    figure.colorbar(image, ax=axis, label=DISPLAY_NAMES[metric])
    figure.tight_layout()
    figure.savefig(output_dir / f"per_series_{metric}.png", dpi=180)
    plt.close(figure)


def save_threshold_diagnostics(plt, test_summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    panels = [
        ("point_f1", "point_best_f1_oracle", "Point F1"),
        ("event_f1", "event_best_f1_oracle", "Event F1"),
    ]
    for axis, (selected, oracle, title) in zip(axes, panels, strict=True):
        frame = test_summary.sort_values(selected, ascending=False).set_index("method")
        frame[[selected, oracle]].plot(
            kind="bar",
            ax=axis,
            color=["#2878B5", "#E45756"],
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("F1" if axis is axes[0] else "")
        axis.set_ylim(0, 1)
        axis.legend(["Selected threshold", "Label-informed oracle diagnostic"])
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("Selected-threshold performance and oracle diagnostic")
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_diagnostics.png", dpi=180)
    plt.close(figure)


def save_training_loss(plt, run_dir: Path, output_dir: Path) -> bool:
    loss_path = run_dir / "ts2vec" / "training" / "loss_log.npy"
    if not loss_path.is_file():
        return False
    loss = np.load(loss_path)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(np.arange(1, len(loss) + 1), loss, color="#2878B5", linewidth=1)
    axis.set_title("TS2Vec training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    figure.tight_layout()
    figure.savefig(output_dir / "ts2vec_training_loss.png", dpi=180)
    plt.close(figure)
    return True


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number"):
        if column in {"series_count", "false_positive_events"}:
            selected[column] = selected[column].map(lambda value: f"{value:.0f}")
        else:
            selected[column] = selected[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    run_dir: Path,
    output_dir: Path,
    inventory: pd.DataFrame,
    test_summary: pd.DataFrame,
    statistics: pd.DataFrame,
    has_loss_plot: bool,
) -> None:
    ranked = test_summary.sort_values("point_f1", ascending=False)
    leaders = {
        metric: test_summary.loc[test_summary[metric].idxmax()]
        for metric in SUMMARY_METRICS
    }
    incomplete_count = int((~inventory["complete"]).sum()) if not inventory.empty else 0
    figure_names = [
        "test_metric_comparison.png",
        "precision_recall.png",
        "per_series_point_f1.png",
        "per_series_event_f1.png",
        "threshold_diagnostics.png",
    ]
    if has_loss_plot:
        figure_names.append("ts2vec_training_loss.png")

    lines = [
        f"# Results Analysis: {run_dir.name}",
        "",
        "## Scope",
        "",
        f"- Analyzed run: `{run_dir}`",
        f"- Models in test summary: {len(test_summary)}",
        f"- Result directories found: {len(inventory)}",
        f"- Directories without both baseline and TS2Vec summaries: {incomplete_count}",
        "",
        "## Aggregate Test Ranking",
        "",
        markdown_table(
            ranked,
            [
                "method",
                "point_f1",
                "point_pr_auc",
                "event_f1",
                "point_precision",
                "point_recall",
                "event_precision",
                "event_recall",
            ],
        ),
        "",
        "## Main Observations",
        "",
        (
            f"- Best point F1: **{leaders['point_f1']['method']}** "
            f"({leaders['point_f1']['point_f1']:.3f})."
        ),
        (
            f"- Best point PR AUC: **{leaders['point_pr_auc']['method']}** "
            f"({leaders['point_pr_auc']['point_pr_auc']:.3f})."
        ),
        (
            f"- Best event F1: **{leaders['event_f1']['method']}** "
            f"({leaders['event_f1']['event_f1']:.3f})."
        ),
        (
            "- Event F1 is substantially lower than pointwise performance for most methods. "
            "The per-series tables show that many detected event segments are false positives."
        ),
        (
            "- Oracle values use labels and are diagnostic only. They are shown for context and "
            "must not be reported as final model performance."
        ),
        "",
        "## Per-Series Descriptive Statistics",
        "",
        markdown_table(
            statistics,
            [
                "method",
                "series_count",
                "point_f1_mean",
                "point_f1_std",
                "event_f1_mean",
                "event_f1_std",
                "false_positive_events",
            ],
        ),
        "",
        "## Generated Figures",
        "",
        *[
            item
            for name in figure_names
            for item in (f"### {Path(name).stem.replace('_', ' ').title()}", "", f"![{name}]({name})", "")
        ],
        "",
        "Reported event metrics use NAB windows but are not the official NAB score.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir, inventory = select_run(args.results_dir, args.run_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, per_series = load_run_results(run_dir)
    test_summary = summary[summary["split"] == "test"].copy()
    per_series_test = per_series[per_series["split"] == "test"].copy()
    if test_summary.empty or per_series_test.empty:
        raise SystemExit(f"Run has no test metrics: {run_dir}")

    ranking = test_summary.sort_values(
        ["point_f1", "event_f1"],
        ascending=False,
    )
    statistics = descriptive_statistics(per_series_test)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    ranking.to_csv(output_dir / "test_model_ranking.csv", index=False)
    statistics.to_csv(output_dir / "per_series_statistics.csv", index=False)
    (output_dir / "oracle_gap.png").unlink(missing_ok=True)

    plt = configure_matplotlib(output_dir)
    save_metric_comparison(plt, test_summary, output_dir)
    save_precision_recall(plt, test_summary, output_dir)
    save_heatmap(plt, per_series_test, "point_f1", output_dir)
    save_heatmap(plt, per_series_test, "event_f1", output_dir)
    save_threshold_diagnostics(plt, test_summary, output_dir)
    has_loss_plot = save_training_loss(plt, run_dir, output_dir)
    write_report(
        run_dir,
        output_dir,
        inventory,
        test_summary,
        statistics,
        has_loss_plot,
    )

    print(f"Analyzed run: {run_dir}")
    print(f"Models: {', '.join(ranking['method'])}")
    print(f"Outputs: {output_dir}")
    print("\nTest ranking:")
    print(
        ranking[
            ["method", "point_f1", "point_pr_auc", "event_f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

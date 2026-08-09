"""Aggregate seed-level uncertainty and produce the MVP figures."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "negs-report-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _confidence_interval(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, 0.0
    critical = _T_CRITICAL_95.get(len(clean) - 1, 1.96)
    half_width = critical * float(clean.std(ddof=1)) / math.sqrt(len(clean))
    return mean, half_width


def _summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, n), group in seed_frame.groupby(["method", "n"], sort=True):
        ratio_mean, ratio_ci = _confidence_interval(group["optimality_ratio"])
        exact_mean, exact_ci = _confidence_interval(group["exact_optimum"])
        records.append(
            {
                "method": method,
                "n": int(n),
                "seed_count": int(group["seed"].nunique()),
                "optimality_ratio_mean": ratio_mean,
                "optimality_ratio_ci95": ratio_ci,
                "exact_optimum_rate_mean": exact_mean,
                "exact_optimum_rate_ci95": exact_ci,
                "mean_inference_time_seconds": float(group["inference_time_seconds"].mean()),
                **(
                    {"parameter_count": int(group["parameter_count"].max())}
                    if "parameter_count" in group
                    else {}
                ),
                "constraint_violations": int(group["constraint_violations"].sum()),
                "terminal_maximal_rate": float(group["terminal_maximal"].mean()),
            }
        )
    return pd.DataFrame(records)


def _plot_metric(
    summary: pd.DataFrame,
    *,
    mean_column: str,
    ci_column: str,
    ylabel: str,
    output: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    preferred_order = [
        "turan_oracle",
        "gnn",
        "mlp",
        "least_degree",
        "random",
        "untrained_gnn",
        "untrained_mlp",
    ]
    available = set(summary["method"])
    methods = [method for method in preferred_order if method in available]
    methods.extend(sorted(available - set(methods)))
    palette = sns.color_palette("colorblind", len(methods))
    for color, method in zip(palette, methods, strict=True):
        data = summary[summary["method"] == method].sort_values("n")
        axis.errorbar(
            data["n"],
            data[mean_column],
            yerr=data[ci_column],
            label=method.replace("_", " "),
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            linestyle="--" if method == "turan_oracle" else "-",
        )
    axis.set_xlabel("Graph size n")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(summary["n"].unique()))
    axis.set_ylim(-0.03, 1.03)
    if max(summary["n"]) > 14:
        axis.axvline(15, color="0.45", linestyle=":", linewidth=1.5)
        axis.text(15.25, 0.035, "unseen sizes", color="0.35", fontsize=10)
    axis.legend(
        frameon=True,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_training_curve(run_name: str, output: Path) -> bool:
    run_names = [run_name]
    if run_name.endswith("-v0.2"):
        base_name = run_name.removesuffix("-v0.2")
        run_names = [base_name, f"{base_name}-mlp"]
    records: list[dict[str, float | int | str]] = []
    for training_run_name in run_names:
        family = "mlp" if training_run_name.endswith("-mlp") else "gnn"
        for metrics_path in sorted(Path("runs").glob(f"{training_run_name}-seed-*/metrics.jsonl")):
            seed = int(metrics_path.parent.name.rsplit("-", 1)[-1])
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                value = json.loads(line)
                if "validation_mean_ratio" in value:
                    records.append(
                        {
                            "family": family,
                            "seed": seed,
                            "global_step": int(value["global_step"]),
                            "validation_mean_ratio": float(value["validation_mean_ratio"]),
                        }
                    )
    if not records:
        return False
    frame = pd.DataFrame(records)
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    multiple_families = frame["family"].nunique() > 1
    for (family, seed), group in frame.groupby(["family", "seed"]):
        group = group.sort_values("global_step")
        axis.plot(
            group["global_step"],
            group["validation_mean_ratio"],
            marker="o",
            alpha=0.75,
            label=f"{family} seed {seed}" if multiple_families else f"seed {seed}",
        )
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Mean validation optimality ratio")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=True, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def generate_report(results_path: str | Path) -> Path:
    source = Path(results_path)
    frame = pd.read_csv(source)
    required = {
        "method",
        "n",
        "seed",
        "optimality_ratio",
        "exact_optimum",
        "inference_time_seconds",
        "constraint_violations",
        "terminal_maximal",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"evaluation CSV is missing columns: {sorted(missing)}")

    aggregations = {
        "optimality_ratio": ("optimality_ratio", "mean"),
        "exact_optimum": ("exact_optimum", "mean"),
        "inference_time_seconds": ("inference_time_seconds", "mean"),
        "constraint_violations": ("constraint_violations", "sum"),
        "terminal_maximal": ("terminal_maximal", "mean"),
    }
    if "parameter_count" in frame:
        aggregations["parameter_count"] = ("parameter_count", "first")
    seed_frame = (
        frame.groupby(["method", "n", "seed"], as_index=False)
        .agg(**aggregations)
        .sort_values(["method", "n", "seed"])
    )
    summary = _summary(seed_frame)
    summary_path = source.parent / "summary.csv"
    summary.to_csv(summary_path, index=False)

    figure_dir = Path("reports/figures")
    if source.parent.name != "mvp":
        figure_dir /= source.parent.name
    _plot_metric(
        summary,
        mean_column="optimality_ratio_mean",
        ci_column="optimality_ratio_ci95",
        ylabel="Optimality ratio",
        output=figure_dir / "optimality_ratio.png",
    )
    _plot_metric(
        summary,
        mean_column="exact_optimum_rate_mean",
        ci_column="exact_optimum_rate_ci95",
        ylabel="Exact-optimum success rate",
        output=figure_dir / "exact_optimum_rate.png",
    )
    run_name = source.parent.name
    _plot_training_curve(run_name, figure_dir / "training_curve.png")

    markdown_path = source.parent / "summary.md"
    columns = [
        "method",
        "n",
        "optimality_ratio_mean",
        "optimality_ratio_ci95",
        "exact_optimum_rate_mean",
        "exact_optimum_rate_ci95",
    ]
    if "parameter_count" in summary:
        columns.append("parameter_count")
    selected = summary[columns].copy()
    headers = list(selected.columns)
    rows = [
        [f"{value:.4f}" if isinstance(value, float | np.floating) else str(value) for value in row]
        for row in selected.itertuples(index=False, name=None)
    ]
    table = (
        "| "
        + " | ".join(headers)
        + " |\n"
        + "| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "".join("| " + " | ".join(row) + " |\n" for row in rows)
    )
    markdown_path.write_text(
        "# MVP result summary\n\n"
        "Intervals are 95% Student-t intervals over seed-level means.\n\n" + table,
        encoding="utf-8",
    )
    return summary_path

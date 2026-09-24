"""Regenerate seed-level tables and comparisons for the frozen parity follow-up."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from analyze_study import plt, seed_summary  # noqa: E402

from extremal_graph.utils import sha256_file, write_json  # noqa: E402

STUDY = ROOT / "study/followup"


def main():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    verification = json.loads((STUDY / "verification.json").read_text())
    expected = (
        len(protocol["methods"])
        * len(protocol["training_seeds"])
        * len(protocol["sizes"])
        * protocol["episodes_per_cell"]
    )
    if verification != {
        "cells": expected // protocol["episodes_per_cell"],
        "graphs": expected,
        "valid": True,
    }:
        raise ValueError("complete, verified parity panel required")
    source = STUDY / "artifacts/evaluation.csv"
    frame = pd.read_csv(source, low_memory=False)
    if len(frame) != expected or frame.record_id.duplicated().any():
        raise ValueError("follow-up results are missing or duplicated")
    seed, summary = seed_summary(frame)
    contrasts = []
    for n in protocol["sizes"]:
        panel = seed[seed.n == n].pivot(index="seed", columns="method")
        for control in ["endpoint_parity", "candidate_parity", "balance_parity", "uniform_parity"]:
            for metric in ["exact_optimum", "optimality_ratio", "absolute_gap"]:
                differences = panel[metric]["gnn_parity"] - panel[metric][control]
                if len(differences) != 5 or differences.isna().any():
                    raise ValueError("paired follow-up contrast lacks a seed")
                center = differences.mean()
                half = 2.7764451051977987 * differences.std(ddof=1) / math.sqrt(5)
                contrasts.append(
                    {
                        "n": n,
                        "comparison": "gnn_parity - " + control,
                        "metric": metric,
                        "difference": center,
                        "ci95_low": center - half,
                        "ci95_high": center + half,
                        "seed_differences": json.dumps(differences.tolist()),
                        "planned_role": "primary"
                        if (n == 40 and control == "endpoint_parity" and metric == "exact_optimum")
                        else "secondary_exploratory",
                        "interval_note": (
                            "Five paired seed means; secondary intervals "
                            "unadjusted for multiplicity"
                        ),
                    }
                )
    tables = STUDY / "tables"
    figures = STUDY / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    outputs = {
        "parity_seed_results.csv": seed,
        "parity_summary.csv": summary,
        "parity_comparisons.csv": pd.DataFrame(contrasts),
    }
    for name, table in outputs.items():
        table.to_csv(tables / name, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = {
        "gnn_parity": "GNN + parity",
        "endpoint_parity": "Endpoint + parity",
        "candidate_parity": "Candidate + parity",
        "uniform_parity": "Uniform + parity",
        "balance_parity": "Balance rule + parity",
    }
    for axis, metric in zip(axes, ["optimality_ratio", "exact_optimum"], strict=True):
        for method in protocol["methods"]:
            data = summary[(summary.method == method) & (summary.metric == metric)].sort_values("n")
            axis.errorbar(
                data.n,
                data["mean"],
                yerr=data.ci95_halfwidth,
                marker="o",
                capsize=2,
                label=labels[method],
            )
        axis.set(
            xlabel="Vertices n",
            ylabel="Edge ratio" if metric == "optimality_ratio" else "Exact success",
            ylim=(0.85, 1.02) if metric == "optimality_ratio" else (-0.02, 1.02),
        )
        axis.grid(alpha=0.2)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Frozen scorers under the same explicit bipartite constraint")
    figure = figures / "parity_controls.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    report_inputs = {
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "evaluation_sha256": sha256_file(source),
        "analysis_sha256": sha256_file(__file__),
        "graphs": len(frame),
        "seed_count": 5,
        "tables": {name: sha256_file(tables / name) for name in outputs},
        "figure_sha256": sha256_file(figure),
    }
    write_json(STUDY / "parity_report_inputs.json", report_inputs)
    print(
        summary[summary.metric.isin(["exact_optimum", "optimality_ratio"])].to_string(index=False)
    )
    print(pd.DataFrame(contrasts).query("planned_role == 'primary'").to_string(index=False))


if __name__ == "__main__":
    main()

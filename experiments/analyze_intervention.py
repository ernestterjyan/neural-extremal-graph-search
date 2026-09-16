"""Seed-level uncertainty and plots for the separately declared parity intervention."""

from __future__ import annotations

import json
import math

import pandas as pd
from analyze_study import FIG, OUT, ROOT, plt, seed_summary

from extremal_graph.utils import sha256_file, write_json


def main():
    source = ROOT / "study/artifacts/intervention/evaluation.csv"
    verification = json.loads((ROOT / "study/intervention-verification.json").read_text())
    if verification["graphs"] != 6000 or not verification["valid"]:
        raise ValueError("complete verified intervention required")
    frame = pd.read_csv(source)
    frame["parameter_count"] = frame.method.map(
        {"gnn_fresh": 38337, "gnn_parity": 38337, "uniform_parity": 0}
    )
    seed, summary = seed_summary(frame)
    comparisons = []
    for n in [15, 24, 30, 40]:
        for control in ["gnn_fresh", "uniform_parity"]:
            for metric in ["optimality_ratio", "exact_optimum"]:
                pair = seed[seed.n == n].pivot(index="seed", columns="method", values=metric)
                difference = pair.gnn_parity - pair[control]
                if difference.isna().any() or len(difference) != 5:
                    raise ValueError("missing intervention replicate")
                half = 2.7764451051977987 * difference.std(ddof=1) / math.sqrt(5)
                comparisons.append(
                    dict(
                        n=n,
                        comparison="gnn_parity - " + control,
                        metric=metric,
                        difference=difference.mean(),
                        ci95_low=difference.mean() - half,
                        ci95_high=difference.mean() + half,
                        seed_differences=json.dumps(difference.tolist()),
                        interpretation="exploratory, descriptive unadjusted interval",
                    )
                )
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    seed.to_csv(OUT / "intervention_seed_results.csv", index=False)
    summary.to_csv(OUT / "intervention_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "intervention_comparisons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric in zip(axes, ["optimality_ratio", "exact_optimum"], strict=True):
        for method in ["gnn_fresh", "gnn_parity", "uniform_parity"]:
            data = summary[(summary.method == method) & (summary.metric == metric)].sort_values("n")
            ax.errorbar(
                data.n, data["mean"], yerr=data.ci95_halfwidth, marker="o", label=method, capsize=3
            )
        ax.set(
            xlabel="Vertices n",
            ylabel="Edge ratio" if metric == "optimality_ratio" else "Exact success",
            ylim=(-0.02, 1.025),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Exploratory parity constraint: extra graph knowledge supplied to both protected arms"
    )
    fig.savefig(FIG / "parity_intervention.png", dpi=180)
    plt.close(fig)
    write_json(
        ROOT / "study/intervention_report_inputs.json",
        {
            "evaluation_sha256": sha256_file(source),
            "protocol_sha256": sha256_file(ROOT / "study/intervention-protocol.json"),
            "analysis_sha256": sha256_file(__file__),
            "graphs": len(frame),
            "seed_count": 5,
            "tables": {p.name: sha256_file(p) for p in OUT.glob("intervention_*.csv")},
            "figure_sha256": sha256_file(FIG / "parity_intervention.png"),
        },
    )
    print(
        summary[
            summary.metric.isin(["optimality_ratio", "exact_optimum", "nonbipartite_rate"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

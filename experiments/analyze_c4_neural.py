"""Summarize the frozen exploratory neural C4 panel against the prior controls."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.c4_neural_pilot import ART, STUDY, sha, verify  # noqa: E402


def main() -> None:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    records = verify(protocol)
    frame = pd.DataFrame(records)
    frame["exact"] = frame.apply(
        lambda row: int(row.edge_count == protocol["known_exact_values"][str(row.n)]), axis=1
    )
    metrics = ["edge_count", "exact", "seconds_per_graph"]
    seed = (
        frame.groupby(["n", "family", "training_seed"], as_index=False)[metrics]
        .mean()
        .sort_values(["n", "family", "training_seed"])
    )
    summary = []
    for (n, family), part in seed.groupby(["n", "family"]):
        for metric in metrics:
            values = part[metric]
            center = values.mean()
            half = 4.302652729911275 * values.std(ddof=1) / math.sqrt(3)
            summary.append(
                {
                    "n": n,
                    "family": family,
                    "metric": metric,
                    "mean": center,
                    "minimum_seed_mean": values.min(),
                    "maximum_seed_mean": values.max(),
                    "descriptive_ci95_low": center - half,
                    "descriptive_ci95_high": center + half,
                    "training_seeds": 3,
                }
            )
    summary = pd.DataFrame(summary)
    contrasts = []
    for n in protocol["evaluation_sizes"]:
        panel = seed[seed.n == n].pivot(index="training_seed", columns="family")
        for metric in metrics[:2]:
            differences = panel[metric]["gnn"] - panel[metric]["endpoint"]
            center = differences.mean()
            half = 4.302652729911275 * differences.std(ddof=1) / math.sqrt(3)
            contrasts.append(
                {
                    "n": n,
                    "metric": metric,
                    "gnn_minus_endpoint": center,
                    "descriptive_ci95_low": center - half,
                    "descriptive_ci95_high": center + half,
                    "seed_differences": json.dumps(differences.tolist()),
                    "note": "Exploratory; three paired training seeds; not a superiority test",
                }
            )
    baseline = pd.read_csv(ROOT / "study/c4_pilot/summary.csv")
    reference = baseline[baseline.method.isin(["polarity", "polarity_swap200"])][
        ["n", "method", "mean_edges", "max_edges", "exact_runs", "mean_seconds"]
    ].copy()
    training = []
    hashes = {}
    for family in protocol["families"]:
        for training_seed in protocol["training_seeds"]:
            folder = ART / "training" / f"{family}-seed-{training_seed}"
            import torch

            payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            history = [
                json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
            ]
            training.append(
                {
                    "family": family,
                    "training_seed": training_seed,
                    "training_rollouts": len(history) * protocol["rollouts_per_iteration"],
                    "training_actions": sum(
                        event["mean_train_edges"] * protocol["rollouts_per_iteration"]
                        for event in history
                    ),
                    "optimizer_updates": len(history) * protocol["updates_per_iteration"],
                    "validation_episodes": len(history)
                    // protocol["validation_interval"]
                    * protocol["validation_episodes"],
                    "selected_iteration": payload["best_step"],
                    "selected_validation_mean_edges": payload["best_validation_score"],
                    "parameter_count": payload["parameter_count"],
                    "training_seconds": sum(event["seconds"] for event in history),
                }
            )
            hashes[str((folder / "best.pt").relative_to(ROOT))] = sha(folder / "best.pt")
            hashes[str((folder / "metrics.jsonl").relative_to(ROOT))] = sha(
                folder / "metrics.jsonl"
            )
    tables = STUDY / "tables"
    tables.mkdir(exist_ok=True)
    outputs = {
        "seed_results.csv": seed,
        "summary.csv": summary,
        "paired_contrasts.csv": pd.DataFrame(contrasts),
        "polarity_comparison.csv": reference,
        "training.csv": pd.DataFrame(training),
    }
    for name, data in outputs.items():
        data.to_csv(tables / name, index=False)
    figures = STUDY / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {"gnn": "#1565c0", "endpoint": "#ef6c00"}
    for family in protocol["families"]:
        part = summary[(summary.family == family) & (summary.metric == "edge_count")].sort_values(
            "n"
        )
        axes[0].plot(part.n, part["mean"], marker="o", color=colors[family], label=family)
        axes[0].fill_between(
            part.n, part.minimum_seed_mean, part.maximum_seed_mean, color=colors[family], alpha=0.15
        )
        exact = summary[(summary.family == family) & (summary.metric == "exact")].sort_values("n")
        axes[1].plot(exact.n, exact["mean"], marker="o", color=colors[family], label=family)
        axes[1].fill_between(
            exact.n,
            exact.minimum_seed_mean,
            exact.maximum_seed_mean,
            color=colors[family],
            alpha=0.15,
        )
    polarity = baseline[baseline.method == "polarity"].sort_values("n")
    axes[0].plot(polarity.n, polarity.mean_edges, "k--", marker="s", label="polarity")
    axes[0].plot(polarity.n, polarity.known_exact, "k:", label="published exact")
    axes[1].plot(
        polarity.n, polarity.exact_runs / polarity.replications, "k--", marker="s", label="polarity"
    )
    axes[0].set(xlabel="Vertices n", ylabel="Terminal edges")
    axes[1].set(xlabel="Vertices n", ylabel="Exact-success fraction", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("C4-free construction: new neural policies versus algebraic baseline")
    figure = figures / "solver_comparison.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    report_inputs = {
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "evaluation_sha256": sha(STUDY / "results.jsonl"),
        "baseline_sha256": sha(ROOT / "study/c4_pilot/results.jsonl"),
        "analysis_sha256": sha(Path(__file__)),
        "training_inputs": hashes,
        "graphs": len(records),
        "tables": {name: sha(tables / name) for name in outputs},
        "figure_sha256": sha(figure),
    }
    (STUDY / "report_inputs.json").write_text(json.dumps(report_inputs, indent=2) + "\n")
    print(summary[summary.metric.isin(["edge_count", "exact"])].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()

"""Analyze the separately frozen fixed-size, equal-update training study."""

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
ART = STUDY / "artifacts"


def main():
    protocol = json.loads((STUDY / "matched_protocol.json").read_text())
    verified = json.loads((STUDY / "matched_verification.json").read_text())
    expected_cells = 55
    if verified != {"cells": expected_cells, "graphs": expected_cells * 100, "valid": True}:
        raise ValueError("complete verified matched evaluation required")
    source = ART / "matched_evaluation/evaluation.csv"
    frame = pd.read_csv(source, low_memory=False)
    if len(frame) != expected_cells * 100 or frame.record_id.duplicated().any():
        raise ValueError("matched evaluation is incomplete or duplicated")
    frame["training_seed"] = frame.seed - 301
    if set(frame.training_seed) != set(protocol["training_seeds"]):
        raise ValueError("unexpected evaluation seed keys")
    seed, summary = seed_summary(frame)
    contrasts = []
    for n in protocol["evaluation_sizes"]:
        panel = seed[seed.n == n].pivot(index="seed", columns="method")
        for control in ["endpoint", "candidate", "mlp"]:
            if control not in panel.columns.get_level_values("method"):
                continue
            for metric in ["exact_optimum", "optimality_ratio", "absolute_gap"]:
                differences = panel[metric]["gnn"] - panel[metric][control]
                if len(differences) != 5 or differences.isna().any():
                    raise ValueError("matched contrast lacks a complete seed pair")
                center = differences.mean()
                half = 2.7764451051977987 * differences.std(ddof=1) / math.sqrt(5)
                contrasts.append(
                    {
                        "n": n,
                        "comparison": "gnn - " + control,
                        "metric": metric,
                        "difference": center,
                        "ci95_low": center - half,
                        "ci95_high": center + half,
                        "seed_differences": json.dumps(differences.tolist()),
                        "planned_role": "primary"
                        if (n == 24 and control == "endpoint" and metric == "exact_optimum")
                        else "secondary_exploratory",
                        "interval_note": (
                            "Five paired training seeds; secondary intervals "
                            "unadjusted for multiplicity"
                        ),
                    }
                )
    rows = []
    curves = []
    input_hashes = {}
    for family in protocol["families"]:
        for training_seed in protocol["training_seeds"]:
            folder = ART / "matched_training" / f"{family}-seed-{training_seed}"
            completion_path = folder / "completion.json"
            metrics_path = folder / "metrics.jsonl"
            checkpoint_path = folder / "best.pt"
            completion = json.loads(completion_path.read_text())
            events = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            if len(events) != protocol["iterations"]:
                raise ValueError("incomplete matched training log")
            rows.append(
                {
                    "method": family,
                    "seed": training_seed,
                    "iterations": len(events),
                    "training_rollouts": sum(e["training_rollouts"] for e in events),
                    "training_actions": sum(e["training_actions"] for e in events),
                    "elite_prefix_pairs": sum(e["elite_prefix_pairs"] for e in events),
                    "optimizer_updates": sum(e["optimizer_updates"] for e in events),
                    "optimizer_pair_presentations": sum(
                        e["optimizer_pair_presentations"] for e in events
                    ),
                    "validation_episodes": sum(e["validation_episodes"] for e in events),
                    "selected_step": completion["best_step"],
                    "selected_validation_score": completion["best_validation_score"],
                    "wall_seconds": sum(e["seconds"] for e in events),
                    "checkpoint_sha256": completion["best_sha256"],
                }
            )
            for event in events:
                if event["validation_score"] is not None:
                    curves.append(
                        {
                            "method": family,
                            "seed": training_seed,
                            "step": event["step"],
                            "validation_score": event["validation_score"],
                        }
                    )
            for path in [completion_path, metrics_path, checkpoint_path]:
                input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    training = pd.DataFrame(rows)
    for column, expected in [
        ("training_rollouts", 25_600),
        ("optimizer_updates", 800),
        ("elite_prefix_pairs", 62_400),
        ("optimizer_pair_presentations", 204_800),
        ("validation_episodes", 2_000),
    ]:
        if not (training[column] == expected).all():
            raise ValueError(f"{column} varies across matched runs")
    tables = STUDY / "tables"
    figures = STUDY / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    outputs = {
        "matched_seed_results.csv": seed,
        "matched_summary.csv": summary,
        "matched_comparisons.csv": pd.DataFrame(contrasts),
        "matched_training.csv": training,
        "matched_validation_curves.csv": pd.DataFrame(curves),
    }
    for name, data in outputs.items():
        data.to_csv(tables / name, index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    curve = pd.DataFrame(curves).groupby(["method", "step"], as_index=False).validation_score.mean()
    for family in protocol["families"]:
        data = curve[curve.method == family]
        axes[0].plot(data.step, data.validation_score, marker=".", label=family)
        for axis, metric in [(axes[1], "optimality_ratio"), (axes[2], "exact_optimum")]:
            subset = summary[(summary.method == family) & (summary.metric == metric)].sort_values(
                "n"
            )
            axis.errorbar(
                subset.n,
                subset["mean"],
                yerr=subset.ci95_halfwidth,
                marker="o",
                capsize=2,
                label=family,
            )
    axes[0].set(xlabel="Training iteration", ylabel="Mean n=14 validation ratio", ylim=(0.3, 1.02))
    axes[1].set(xlabel="Vertices n", ylabel="Edge ratio", ylim=(0.35, 1.03))
    axes[2].set(xlabel="Vertices n", ylabel="Exact success", ylim=(-0.02, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[2].legend(fontsize=9)
    fig.suptitle("Fixed n=14 exposure, equal rollout and update counts")
    figure = figures / "matched_exposure.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    write_json(
        STUDY / "matched_report_inputs.json",
        {
            "protocol_sha256": sha256_file(STUDY / "matched_protocol.json"),
            "evaluation_sha256": sha256_file(source),
            "analysis_sha256": sha256_file(__file__),
            "training_inputs": input_hashes,
            "training_runs": len(training),
            "evaluation_graphs": len(frame),
            "tables": {name: sha256_file(tables / name) for name in outputs},
            "figure_sha256": sha256_file(figure),
        },
    )
    print(training.groupby("method")["selected_validation_score"].agg(["mean", "min", "max"]))
    print(
        summary[summary.metric.isin(["exact_optimum", "optimality_ratio"])].to_string(index=False)
    )
    print(pd.DataFrame(contrasts).query("planned_role == 'primary'").to_string(index=False))


if __name__ == "__main__":
    main()

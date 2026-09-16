"""Regenerate all corrected-study tables, plots and stratified evidence references."""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "negs-study-mpl"))
cache = Path(tempfile.gettempdir()) / "negs-study-cache"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from extremal_graph.utils import sha256_file, write_json  # noqa: E402

ART = ROOT / "study/artifacts"
OUT = ROOT / "study/tables"
FIG = ROOT / "study/figures"
FAMILIES = ["gnn", "mlp", "candidate", "endpoint"]


def seed_summary(frame):
    keys = ["method", "n", "seed"]
    work = frame.copy()
    for category in ["exact", "unbalanced_bipartite", "nonbipartite"]:
        work[category + "_rate"] = (work.outcome == category).astype(int)
    for cause in ["odd_cycle", "partition_imbalance"]:
        work[cause + "_first_loss_rate"] = (work.first_loss_cause == cause).astype(int)
    columns = [
        "optimality_ratio",
        "exact_optimum",
        "absolute_gap",
        "inference_time_seconds",
        "first_loss_step",
        "first_odd_cycle_step",
        "parameter_count",
    ]
    columns += [c for c in work if c.endswith("_rate")]
    seed = work.groupby(keys, as_index=False)[columns].mean()
    records = []
    for (method, n), group in seed.groupby(["method", "n"]):
        for metric in columns:
            x = group[metric].dropna()
            count = len(x)
            half = 2.7764451051977987 * x.std(ddof=1) / math.sqrt(count) if count == 5 else np.nan
            records.append(
                dict(
                    method=method,
                    n=n,
                    metric=metric,
                    mean=x.mean(),
                    ci95_halfwidth=half,
                    seed_count=count,
                )
            )
    return seed, pd.DataFrame(records)


def comparisons(seed):
    records = []
    for n in [14, 24, 40]:
        for control in ["mlp", "candidate", "endpoint"]:
            for metric in ["optimality_ratio", "exact_optimum"]:
                pair = seed[seed.n == n].pivot(index="seed", columns="method", values=metric)
                if control not in pair or "gnn" not in pair:
                    continue
                diff = (pair.gnn - pair[control]).dropna()
                if len(diff) != 5:
                    raise ValueError("comparison requires all five seeds")
                half = 2.7764451051977987 * diff.std(ddof=1) / math.sqrt(5)
                records.append(
                    dict(
                        n=n,
                        comparison="gnn - " + control,
                        metric=metric,
                        difference=diff.mean(),
                        ci95_low=diff.mean() - half,
                        ci95_high=diff.mean() + half,
                        seed_differences=json.dumps(diff.tolist()),
                        interpretation="descriptive unadjusted interval; multiple comparisons",
                    )
                )
    return pd.DataFrame(records)


def training_tables():
    rows = []
    validations = []
    iterations = []
    for family in FAMILIES:
        for seed in range(5):
            run = ART / f"training/corrected-{family}-seed-{seed}"
            completion = json.loads((run / "completion.json").read_text())
            metadata = json.loads((run / "metadata.json").read_text())
            metrics = [
                json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            if sha256_file(run / "best.pt") != completion["checkpoint_sha256"]:
                raise ValueError("training checkpoint hash mismatch")
            selected = checkpoint["provenance"]["selected_global_step"]
            selected_record = next(x for x in metrics if x["global_step"] == selected)
            if (
                abs(selected_record["validation_mean_ratio"] - checkpoint["validation_score"])
                > 1e-12
            ):
                raise ValueError("selected validation provenance mismatch")
            budget = metrics[:50]
            record = dict(
                method=family,
                seed=seed,
                iterations=len(metrics),
                rollouts=sum(x["rollout_episodes"] for x in metrics),
                sampled_actions=sum(x["sampled_actions"] for x in metrics),
                optimizer_updates=sum(x["optimizer_updates"] for x in metrics),
                iteration_wall_seconds=sum(x["iteration_wall_seconds"] for x in metrics),
                validation_episodes=sum(len(x.get("validation", {})) * 100 for x in metrics),
                selected_global_step=selected,
                selected_validation_mean=checkpoint["validation_score"],
                selected_n14_validation=selected_record["validation"]["14"],
                last_validation_mean=next(
                    x["validation_mean_ratio"]
                    for x in reversed(metrics)
                    if "validation_mean_ratio" in x
                ),
                stage_gate_passes=sum(
                    x["stage_complete"] and x["stage_gate_passed"] for x in metrics
                ),
                budget_rollouts=sum(x["rollout_episodes"] for x in budget),
                budget_validation_episodes=sum(len(x.get("validation", {})) * 100 for x in budget),
                launch_to_completion_seconds=(
                    completion["completed_at_unix"] - metadata["started_at_unix"]
                ),
                budget_actions=sum(x["sampled_actions"] for x in budget),
                budget_updates=sum(x["optimizer_updates"] for x in budget),
                budget_stage_size=budget[-1]["stage_size"],
                source_sha256=metadata["source"]["sha256"],
                checkpoint_sha256=sha256_file(run / "best.pt"),
            )
            rows.append(record)
            for x in metrics:
                iterations.append(dict(method=family, seed=seed, **x))
                for n, value in x.get("validation", {}).items():
                    validations.append(
                        dict(
                            method=family,
                            seed=seed,
                            global_step=x["global_step"],
                            n=int(n),
                            ratio=value,
                            stage_size=x["stage_size"],
                            stage_complete=x["stage_complete"],
                        )
                    )
    if len({r["source_sha256"] for r in rows}) != 1:
        raise ValueError("training cohort source mismatch")
    return pd.DataFrame(rows), pd.DataFrame(validations), pd.DataFrame(iterations)


def probe_tables():
    probes = []
    examples = {}
    for path in sorted((ART / "evaluation").glob("*/seed-*/n-*/chunk-*.json.gz")):
        with gzip.open(path, "rt") as handle:
            records = json.load(handle)["records"]
        for record in records:
            row = record["row"]
            key = (row["method"], row["n"], row["seed"], row["outcome"])
            if key not in examples:
                examples[key] = dict(
                    record_id=row["record_id"],
                    graph_sha256=row["graph_sha256"],
                    graph_reference=str(path.relative_to(ROOT)),
                    outcome=row["outcome"],
                    method=row["method"],
                    n=row["n"],
                    seed=row["seed"],
                    episode=row["episode"],
                )
            if record["probe"] is not None:
                probe = {k: v for k, v in record["probe"].items() if k != "decisions"}
                probe["trajectory_has_mixed_group"] = int(probe["mixed_group_states"] > 0)
                probe["trajectory_has_unavoidable_alias"] = int(
                    probe["unavoidable_alias_states"] > 0
                )
                if row["method"] in {"mlp", "untrained_mlp"}:
                    # Raw structural accumulators mean not measured for the MLP.
                    for field in [
                        "mixed_group_states",
                        "unavoidable_alias_states",
                        "sum_bad_mass_in_mixed_groups",
                        "chosen_bad_in_mixed_group",
                        "trajectory_has_mixed_group",
                        "trajectory_has_unavoidable_alias",
                    ]:
                        probe[field] = np.nan
                probes.append(
                    dict(
                        method=row["method"],
                        n=row["n"],
                        seed=row["seed"],
                        episode=row["episode"],
                        **probe,
                    )
                )
    return pd.DataFrame(probes), pd.DataFrame(examples.values())


def figures(summary, frame, training, validations, iterations, budget):
    plt.rcParams.update({"font.size": 10})
    colors = dict(
        zip(
            [
                "gnn",
                "mlp",
                "candidate",
                "endpoint",
                "random",
                "least_degree",
                "lookahead",
                "untrained_gnn",
                "untrained_mlp",
                "turan_oracle",
            ],
            plt.get_cmap("tab10").colors,
            strict=True,
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for row, methods in enumerate(
        [
            FAMILIES,
            [
                "random",
                "least_degree",
                "lookahead",
                "untrained_gnn",
                "untrained_mlp",
                "turan_oracle",
            ],
        ]
    ):
        for col, metric in enumerate(["optimality_ratio", "exact_optimum"]):
            ax = axes[row, col]
            for method in methods:
                sub = summary[(summary.method == method) & (summary.metric == metric)].sort_values(
                    "n"
                )
                ax.errorbar(
                    sub.n,
                    sub["mean"],
                    yerr=sub.ci95_halfwidth,
                    marker="o" if method == "lookahead" else ".",
                    linestyle="--"
                    if method == "turan_oracle"
                    else "none"
                    if method == "lookahead"
                    else "-",
                    lw=1.5,
                    zorder=4 if method == "lookahead" else 2,
                    label=method,
                    color=colors[method],
                    capsize=2,
                )
            ax.set(
                xlabel="Vertices n",
                ylabel="Mean edge ratio" if col == 0 else "Exact success rate",
                ylim=((0.78, 1.025) if row == 0 else (0.45, 1.025))
                if col == 0
                else (-0.025, 1.025),
            )
            if row == 1:
                ax.text(18, 0.965, "look-ahead and oracle: 1.00 throughout", fontsize=8)
            ax.axvline(14.5, c="gray", ls=":", lw=1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Corrected study: seed means and descriptive 95% t intervals (five seeds)")
    fig.savefig(FIG / "transfer.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    for ax, family in zip(axes, FAMILIES, strict=True):
        sub = frame[frame.method == family]
        cats = pd.crosstab(sub.n, sub.outcome, normalize="index").reindex(
            columns=["exact", "unbalanced_bipartite", "nonbipartite"], fill_value=0
        )
        ax.stackplot(
            cats.index,
            *[cats[c] for c in cats],
            labels=cats.columns,
            colors=["#21918c", "#f3ba53", "#b34b66"],
        )
        ax.set(title=family, xlabel="Vertices n", ylim=(0, 1))
    axes[0].set_ylabel("Fraction of final graphs")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.savefig(FIG / "failure_categories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for col, family in enumerate(FAMILIES):
        for row, n in enumerate([6, 14]):
            ax = axes[row, col]
            for seed in range(5):
                sub = validations[
                    (validations.method == family)
                    & (validations.seed == seed)
                    & (validations.n == n)
                ]
                ax.plot(sub.global_step, sub.ratio, lw=1, label=f"seed {seed}")
                transitions = iterations[
                    (iterations.method == family)
                    & (iterations.seed == seed)
                    & iterations.stage_complete
                ]
                # Each seed has its own transition schedule; dots show transition iterations.
                for _, event in transitions.iterrows():
                    ax.plot(
                        event.global_step, 0.845 + 0.006 * seed, "|", c=plt.get_cmap("tab10")(seed)
                    )
            ax.set(title=f"{family}: fixed n={n}", xlabel="Training iteration", ylim=(0.84, 1.015))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("Validation edge ratio")
    fig.suptitle(
        "Fixed-size validation curves; colored ticks mark each seed’s curriculum transitions"
    )
    fig.savefig(FIG / "training_fixed_size.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ["optimality_ratio", "exact_optimum"], strict=True):
        for j, family in enumerate(FAMILIES):
            final = frame[(frame.method == family) & (frame.n == 24)].groupby("seed")[metric].mean()
            fixed = budget[(budget.method == family) & (budget.n == 24)].set_index("seed")[metric]
            for seed in range(5):
                ax.plot(
                    [j - 0.15, j + 0.15],
                    [fixed.loc[seed], final.loc[seed]],
                    "o-",
                    c=colors[family],
                    alpha=0.6,
                    ms=4,
                )
        ax.set_xticks(range(4), FAMILIES)
        ax.set(
            ylabel="Mean edge ratio" if metric == "optimality_ratio" else "Exact success rate",
            ylim=(0.45, 1.01) if metric == "optimality_ratio" else (-0.02, 1.01),
        )
    fig.suptitle(
        "n=24: each seed at 12,800 training rollouts (left) and curriculum completion (right)"
    )
    fig.savefig(FIG / "budget_comparison.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    verification = json.loads((ROOT / "study/verification.json").read_text())
    if not all(x["valid"] for x in verification.values()):
        raise ValueError("verification is required before reporting")
    frame = pd.read_csv(ART / "evaluation.csv", low_memory=False)
    budget_frame = pd.read_csv(ART / "budget.csv")
    seed, summary = seed_summary(frame)
    bseed, bsummary = seed_summary(budget_frame)
    training, validation, iterations = training_tables()
    probes, examples = probe_tables()
    tables = {
        "seed_results": seed,
        "summary": summary,
        "comparisons": comparisons(seed),
        "budget_seed_results": bseed,
        "budget_summary": bsummary,
        "budget_comparisons": comparisons(bseed),
        "training": training,
        "validation": validation,
        "probes": probes,
        "stratified_examples": examples,
        "outcome_quality": frame.groupby(["method", "n", "seed", "outcome"], as_index=False).agg(
            episodes=("episode", "size"),
            mean_ratio=("optimality_ratio", "mean"),
            mean_gap=("absolute_gap", "mean"),
        ),
    }
    if not probes.empty:
        probe_seed = probes.groupby(["method", "n", "seed"], as_index=False).mean(numeric_only=True)
        tables["probe_seed_results"] = probe_seed
        tables["probe_summary"] = probe_seed.groupby(["method", "n"], as_index=False).mean(
            numeric_only=True
        )
    for name, table in tables.items():
        table.to_csv(OUT / (name + ".csv"), index=False)
    figures(summary, frame, training, validation, iterations, bseed)
    sources = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in [ART / "evaluation.csv", ART / "budget.csv", ROOT / "study/protocol.json"]
    }
    write_json(
        ROOT / "study/report_inputs.json",
        {
            "inputs": sources,
            "training_inputs": {
                str(p.relative_to(ROOT)): sha256_file(p)
                for run in (ART / "training").glob("corrected-*-seed-*")
                for p in [
                    run / name
                    for name in [
                        "metrics.jsonl",
                        "metadata.json",
                        "completion.json",
                        "best.pt",
                        "budget-0050.pt",
                    ]
                ]
            },
            "analysis_script_sha256": sha256_file(__file__),
            "graphs": len(frame),
            "budget_graphs": len(budget_frame),
            "training_runs": len(training),
            "failures": len(list(ART.glob("**/failure.json"))),
            "generated_tables": {
                name + ".csv": sha256_file(OUT / (name + ".csv")) for name in tables
            },
            "generated_figures": {
                name: sha256_file(FIG / name)
                for name in [
                    "transfer.png",
                    "training_fixed_size.png",
                    "failure_categories.png",
                    "budget_comparison.png",
                ]
            },
        },
    )
    print(
        summary[
            (summary.n.isin([14, 24, 40]))
            & summary.metric.isin(["optimality_ratio", "exact_optimum"])
        ].to_string(index=False)
    )
    print(
        training.groupby("method")[
            ["iterations", "rollouts", "sampled_actions", "optimizer_updates", "stage_gate_passes"]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()

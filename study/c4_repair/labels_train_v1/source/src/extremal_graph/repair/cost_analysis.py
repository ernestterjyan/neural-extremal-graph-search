"""Offline costs and a prospectively declared inference-only amortization scenario."""

from __future__ import annotations

import math
import statistics
from collections import Counter


def amortization(
    training_seconds: float, shared_seconds: float, saving_per_decision: float
) -> dict:
    if any(not math.isfinite(v) or v < 0 for v in [training_seconds, shared_seconds]):
        raise ValueError("finite nonnegative offline costs required")
    if not math.isfinite(saving_per_decision):
        raise ValueError("finite latency difference required")
    saving_per_search = 30 * saving_per_decision
    return {
        "decisions_per_search_scenario": 30,
        "seconds_saved_per_decision": saving_per_decision,
        "seconds_saved_per_search_scenario": saving_per_search,
        "model_training_seconds": training_seconds,
        "shared_label_and_tuning_seconds": shared_seconds,
        "searches_to_recover_model_training_only": (
            math.ceil(training_seconds / saving_per_search) if saving_per_search > 0 else None
        ),
        "searches_to_recover_training_labels_and_tuning": (
            math.ceil((training_seconds + shared_seconds) / saving_per_search)
            if saving_per_search > 0
            else None
        ),
        "finite_inference_only_break_even": saving_per_search > 0,
        "interpretation": (
            "conditional cached-selection scenario; not demonstrated end-to-end search speedup"
            if saving_per_search > 0
            else (
                "no finite recovery through faster selection; "
                "learned selection has no measured speed advantage"
            )
        ),
    }


def analyze_costs(models: list[dict], ledger: list[dict], profile: list[dict]) -> dict:
    if (
        len(models) != 16
        or len({(m["family"], m["seed"]) for m in models}) != 16
        or Counter(m["family"] for m in models) != {"gnn": 8, "mlp": 8}
    ):
        raise ValueError("complete sixteen-model cost panel required")
    if {r["n"] for r in profile} != {20, 35, 40} or len(profile) != 3:
        raise ValueError("complete matched development latency profile required")
    stages = [r["stage"] for r in ledger]
    if len(stages) != len(set(stages)):
        raise ValueError("duplicate ledger stage would double-count cost")
    for row in ledger:
        for key in ["paid_eur", "local_wall_seconds"]:
            value = row.get(key, 0)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative ledger costs required")
    label_seconds = sum(
        r.get("local_wall_seconds", 0) for r in ledger if r["stage"].startswith("labels_")
    )
    tuning_seconds = sum(
        r.get("local_wall_seconds", 0) for r in ledger if r["stage"].startswith("tuning_")
    )
    calibration_seconds = sum(
        r.get("local_wall_seconds", 0)
        for r in ledger
        if r["stage"].startswith("development_calibration_")
    )
    adaptive_latency = statistics.mean(
        statistics.median(
            next(c for c in row["conditions"] if c["family"] == "adaptive")["selection_seconds"]
        )
        for row in profile
    )
    by_model = []
    for model in models:
        latency = statistics.mean(
            statistics.median(
                next(
                    c
                    for c in row["conditions"]
                    if c["family"] == model["family"] and c.get("seed") == model["seed"]
                )["selection_seconds"]
            )
            for row in profile
        )
        by_model.append(
            {
                "family": model["family"],
                "seed": model["seed"],
                "mean_of_three_graph_median_selection_seconds": latency,
                "adaptive_mean_of_three_graph_median_selection_seconds": adaptive_latency,
                **amortization(
                    model["elapsed_seconds"],
                    label_seconds + tuning_seconds,
                    adaptive_latency - latency,
                ),
            }
        )
    return {
        "paid_eur_recorded": sum(r["paid_eur"] for r in ledger),
        "local_seconds_recorded": sum(r.get("local_wall_seconds", 0) for r in ledger),
        "label_collection_seconds": label_seconds,
        "classical_tuning_seconds": tuning_seconds,
        "calibration_seconds": calibration_seconds,
        "model_training_seconds": sum(m["elapsed_seconds"] for m in models),
        "training_seconds_by_family": {
            f: sum(m["elapsed_seconds"] for m in models if m["family"] == f) for f in ["gnn", "mlp"]
        },
        "model_scenarios": by_model,
        "scope": "predeclared cached-selection scenario on n20/35/40 and 30 decisions per search",
        "limitations": [
            (
                "Includes feature extraction and selection, excluding checkpoint loading "
                "and common candidate generation."
            ),
            (
                "Search trajectories can use different numbers of repairs; "
                "this scenario is not an observed end-to-end saving."
            ),
            (
                "All shared label/tuning cost is conservatively charged to each deployment "
                "scenario; scenarios must not be added together."
            ),
            (
                "Primary fixed-time graph quality and gate results are analyzed separately; "
                "latency cannot replace them."
            ),
            "A null recovery count means no finite inference-only break-even was demonstrated.",
            (
                "The ledger reports retained experiment/model wall time, "
                "not human research time or all orchestration overhead."
            ),
        ],
    }

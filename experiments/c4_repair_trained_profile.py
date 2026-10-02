"""Matched cached-selection timing; trained GNN/MLP versus adaptive selection."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from extremal_graph.repair.campaign import checkpoint_manifest
from extremal_graph.repair.constructions import starting_pool
from extremal_graph.repair.controls import AdaptiveRegions, ExperimentLease
from extremal_graph.repair.development import load_tuning
from extremal_graph.repair.evidence import append_record, prepare_batch, verify_contract
from extremal_graph.repair.regions import candidate_regions
from extremal_graph.repair.selectors import model_selector
from extremal_graph.repair.training import load_model


def profile(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = checkpoint_manifest(args.checkpoints)
    recipe = verify_contract(args.checkpoints)["protocol"]["repair_configuration"]
    config = load_tuning(args.configuration)
    if (
        list(config.region_sizes) != recipe["region_sizes"]
        or config.per_family != recipe["per_family"]
        or config.repair_seconds != recipe["repair_seconds"]
    ):
        raise ValueError("profile must use the trained common repair recipe")
    protocol = {
        "stage": "development_trained_selector_profile",
        "sizes": [20, 35, 40],
        "seed_offset": 12000000,
        "region_sizes": recipe["region_sizes"],
        "per_family": recipe["per_family"],
        "models": models,
        "configuration": asdict(config),
        "repeats": 20,
        "warmups": 1,
        "condition_order": "seeded shuffle using graph seed + 200000000",
        "amortization_decisions_per_search": 30,
        "scope": (
            "cached selection on identical pools; conditional latency scenario, not search speedup"
        ),
    }
    contract = prepare_batch(args.output, protocol)
    for n in protocol["sizes"]:
        started = time.perf_counter()
        seed = protocol["seed_offset"] + n
        graph = max(starting_pool(n, seed), key=lambda s: s.graph.m).graph
        regions = candidate_regions(
            graph,
            random.Random(seed + 100000000),
            sizes=tuple(recipe["region_sizes"]),
            per_family=recipe["per_family"],
        )
        conditions = [*models, {"family": "adaptive", "block": None}]
        random.Random(seed + 200000000).shuffle(conditions)
        outcomes = []
        for position, condition in enumerate(conditions):
            setup = time.perf_counter()
            family = condition["family"]
            selector = None
            if family != "adaptive":
                path = args.checkpoints / family / str(condition["seed"]) / "best.pt"
                selector = model_selector(load_model(path, condition["checkpoint_sha256"]))
            adaptive = AdaptiveRegions(
                exploration=config.adaptive_exploration,
                smoothing=config.adaptive_smoothing,
                reward_scale=config.adaptive_reward_scale,
            )
            rng = random.Random(seed + 300000000 + position)
            initialization = time.perf_counter() - setup

            def choose(selector=selector, rng=rng, adaptive=adaptive, graph=graph, regions=regions):
                if selector is None:
                    return adaptive.choose(regions, rng), None
                scores = selector(graph, regions)
                highest = max(scores)
                selected = rng.choice(
                    [r for r, s in zip(regions, scores, strict=True) if s == highest]
                )
                return selected, scores

            choose()  # One warmup outside repeated selection timing.
            timings, selected_indices = [], []
            for _ in range(protocol["repeats"]):
                before = time.perf_counter()
                chosen, scores = choose()
                timings.append(time.perf_counter() - before)
                selected_indices.append(regions.index(chosen))
            outcome = {
                **condition,
                "position": position,
                "initialization_seconds": initialization,
                "selection_seconds": timings,
                "selected_indices": selected_indices,
            }
            if scores is not None:
                outcome["last_scores"] = scores
            outcomes.append(outcome)
        append_record(
            args.output / "results.jsonl",
            {
                "n": n,
                "seed": seed,
                "status": "complete",
                "initial_edges": graph.edges(),
                "initial_edge_count": graph.m,
                "repairs": [],
                "regions": [{"vertices": list(r.vertices), "family": r.family} for r in regions],
                "conditions": outcomes,
                "elapsed_seconds": time.perf_counter() - started,
                "source_sha256": contract["source_sha256"],
                "protocol_sha256": contract["protocol_sha256"],
            },
        )
        print(n, len(outcomes), "complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ExperimentLease(stage="trained_selector_profile", batch=args.output):
        profile(args)


if __name__ == "__main__":
    main()

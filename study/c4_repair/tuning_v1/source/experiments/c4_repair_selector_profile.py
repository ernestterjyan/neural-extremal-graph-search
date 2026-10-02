"""Profile untrained selector overhead on development graphs; no performance claim."""

from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import time
from pathlib import Path

import torch

from extremal_graph.repair.constructions import starting_pool
from extremal_graph.repair.controls import ExperimentLease
from extremal_graph.repair.evidence import append_record, prepare_batch, require_development
from extremal_graph.repair.graph import verify_witness
from extremal_graph.repair.regions import candidate_regions
from extremal_graph.repair.selectors import RegionGNN, RegionMLP, model_selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 35, 40])
    parser.add_argument("--seed", type=int, default=11000000)
    args = parser.parse_args()
    require_development(args.sizes)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = {
        "stage": "development_selector_profile",
        "sizes": args.sizes,
        "seed_offset": args.seed,
        "region_sizes": [3, 4, 5],
        "per_family": 4,
        "repeats": 20,
        "models": "untrained random initialization; timing only",
    }
    with ExperimentLease(stage="selector_profile", batch=args.output):
        contract = prepare_batch(args.output, protocol)
        for n in args.sizes:
            seed = args.seed + n
            started = time.perf_counter()
            graph = max(starting_pool(n, seed), key=lambda s: s.graph.m).graph
            candidate_started = time.perf_counter()
            regions = candidate_regions(
                graph, random.Random(seed + 100000000), sizes=(3, 4, 5), per_family=4
            )
            candidate_seconds = time.perf_counter() - candidate_started
            models = []
            for model_type in [RegionGNN, RegionMLP]:
                torch.manual_seed(seed)
                setup = time.perf_counter()
                model = model_type()
                selector = model_selector(model)
                initialization = time.perf_counter() - setup
                parameter_digest = hashlib.sha256()
                for name, parameter in model.state_dict().items():
                    parameter_digest.update(name.encode())
                    parameter_digest.update(parameter.numpy().tobytes())
                selector(graph, regions)  # One warmup, recorded separately from repeated timings.
                timings = []
                for _ in range(protocol["repeats"]):
                    before = time.perf_counter()
                    scores = selector(graph, regions)
                    timings.append(time.perf_counter() - before)
                models.append(
                    {
                        "family": model.family,
                        "architecture": model.architecture,
                        "untrained": True,
                        "parameters_sha256": parameter_digest.hexdigest(),
                        "initialization_seconds": initialization,
                        "median_inference_seconds": statistics.median(timings),
                        "maximum_inference_seconds": max(timings),
                        "inference_seconds": timings,
                        "scores": scores,
                    }
                )
            verify_witness(n, graph.edges())
            append_record(
                args.output / "results.jsonl",
                {
                    "n": n,
                    "seed": seed,
                    "status": "complete",
                    "initial_edges": graph.edges(),
                    "initial_edge_count": graph.m,
                    "repairs": [],
                    "regions": [
                        {"vertices": list(r.vertices), "family": r.family} for r in regions
                    ],
                    "candidate_generation_seconds": candidate_seconds,
                    "models": models,
                    "elapsed_seconds": time.perf_counter() - started,
                    "source_sha256": contract["source_sha256"],
                    "protocol_sha256": contract["protocol_sha256"],
                },
            )
            print(n, [(m["family"], m["median_inference_seconds"]) for m in models], flush=True)


if __name__ == "__main__":
    main()

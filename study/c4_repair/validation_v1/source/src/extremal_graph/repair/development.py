"""Frozen paired development tuning for both designated classical comparators."""

from __future__ import annotations

import json
import random
import statistics
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .evidence import (
    append_record,
    independent_audit,
    json_sha,
    load_records,
    prepare_batch,
    prepare_or_resume,
    require_development,
    sha,
    source_manifest,
    verify_contract,
)
from .search import SearchConfig, neighborhood_search, tabu_search


def trial_configurations(base: SearchConfig) -> list[dict]:
    recipes = {
        "tabu/fixed_h5": (
            "tabu",
            {
                "tabu_tenure_min": 5,
                "tabu_tenure_max": 5,
                "tabu_adaptive_penalty": False,
                "tabu_restart_iterations": 1000,
            },
        ),
        "tabu/reactive_h5": (
            "tabu",
            {"tabu_tenure_min": 5, "tabu_tenure_max": 5, "tabu_restart_iterations": 1000},
        ),
        "tabu/reactive_h7_15": ("tabu", {}),
        "tabu/reactive_h12_25": (
            "tabu",
            {
                "tabu_tenure_min": 12,
                "tabu_tenure_max": 25,
                "tabu_penalty": 0.5,
                "tabu_restart_iterations": 1000,
            },
        ),
        "adaptive/stagnation5": ("adaptive", {"stagnation_repairs": 5}),
        "adaptive/stagnation10": ("adaptive", {}),
        "adaptive/explore_stagnation10": ("adaptive", {"adaptive_exploration": 0.4}),
        "adaptive/stagnation20": ("adaptive", {"stagnation_repairs": 20}),
    }
    return [
        {"id": identifier, "method": method, "configuration": asdict(replace(base, **changes))}
        for identifier, (method, changes) in recipes.items()
    ]


def prepare_tuning(args) -> None:
    from .campaign import require_feasibility

    require_feasibility(args.feasibility)
    require_development(args.sizes)
    gate = verify_contract(args.feasibility)["protocol"]
    base = SearchConfig(
        seconds=60,
        region_sizes=tuple(gate["region_sizes"]),
        per_family=gate["per_family"],
        repair_seconds=gate["repair_seconds"],
    )
    protocol = {
        "stage": "development_tuning",
        "sizes": args.sizes,
        "replications": 3,
        "seed_offset": args.seed,
        "trials": trial_configurations(base),
        "trial_order": "seeded shuffle per size/replication using search seed + 500000000",
        "selection": "largest mean edges at 60s, then 10s, then 1s, then lexicographic trial id",
        "feasibility_results_sha256": sha(args.feasibility / "results.jsonl"),
        "paid_compute_eur": 0,
    }
    prepare_batch(args.output, protocol)
    print(f"Frozen {len(args.sizes) * 3 * 8} paired development searches: {args.output}")


def tune(args) -> None:
    torch.set_num_threads(1)
    contract = verify_contract(args.batch)
    protocol = contract["protocol"]
    require_development(protocol["sizes"])
    if protocol["stage"] != "development_tuning" or contract["source_sha256"] != json_sha(
        source_manifest()
    ):
        raise ValueError("tuning requires its frozen source and protocol")
    prepare_or_resume(args.batch, protocol)
    path = args.batch / "results.jsonl"
    done = (
        {(r["n"], r["replication"], r["trial"]) for r in load_records(path)}
        if path.exists()
        else set()
    )
    for n in protocol["sizes"]:
        for replication in range(protocol["replications"]):
            seed = protocol["seed_offset"] + n * 10000 + replication
            order = list(protocol["trials"])
            random.Random(seed + 500000000).shuffle(order)
            for position, trial in enumerate(order):
                if (n, replication, trial["id"]) in done:
                    continue
                config = SearchConfig(
                    **{
                        **trial["configuration"],
                        "region_sizes": tuple(trial["configuration"]["region_sizes"]),
                    }
                )
                record = {
                    "n": n,
                    "replication": replication,
                    "seed": seed,
                    "trial": trial["id"],
                    "trial_position": position,
                    "method": trial["method"],
                    "status": "complete",
                    "source_sha256": contract["source_sha256"],
                    "protocol_sha256": contract["protocol_sha256"],
                }
                try:
                    result = (
                        tabu_search(n, seed, config)
                        if trial["method"] == "tabu"
                        else neighborhood_search(n, seed, config)
                    )
                    record.update(result)
                except Exception:
                    record.update(status="failed", error=traceback.format_exc())
                append_record(path, record)
                print(
                    n,
                    replication,
                    trial["id"],
                    record.get("edge_count"),
                    record["status"],
                    flush=True,
                )


def selection(protocol: dict, records: list[dict]) -> dict:
    expected = {
        (n, i, t["id"])
        for n in protocol["sizes"]
        for i in range(protocol["replications"])
        for t in protocol["trials"]
    }
    found = {(r["n"], r["replication"], r["trial"]) for r in records}
    if (
        len(records) != len(expected)
        or found != expected
        or any(r["status"] != "complete" for r in records)
    ):
        raise ValueError("complete tuning panel required; failures cannot be silently excluded")
    scores = []
    for trial in protocol["trials"]:
        rows = [r for r in records if r["trial"] == trial["id"]]
        scores.append(
            {
                **trial,
                "searches": len(rows),
                "mean_edges": {
                    str(t): statistics.mean(r["checkpoints"][str(t)]["edges"] for r in rows)
                    for t in [1, 10, 60]
                },
                "mean_edges_by_size": {
                    str(n): statistics.mean(
                        r["checkpoints"]["60"]["edges"] for r in rows if r["n"] == n
                    )
                    for n in protocol["sizes"]
                },
            }
        )
    winners = {}
    for method in ["tabu", "adaptive"]:
        winners[method] = min(
            [r for r in scores if r["method"] == method],
            key=lambda r: (
                -r["mean_edges"]["60"],
                -r["mean_edges"]["10"],
                -r["mean_edges"]["1"],
                r["id"],
            ),
        )
    configuration = dict(winners["adaptive"]["configuration"])
    configuration.update(
        {
            key: value
            for key, value in winners["tabu"]["configuration"].items()
            if key.startswith("tabu_")
        }
    )
    return {
        "trials": scores,
        "winners": {m: r["id"] for m, r in winners.items()},
        "selected_configuration": configuration,
        "scope": "development tuning only; not evidence of held-out solver strength",
    }


def report_tuning(args) -> None:
    checked = independent_audit(args.batch)
    if checked["failed_records"]:
        raise ValueError("retain and resolve failed tuning cells before selecting parameters")
    contract = verify_contract(args.batch)
    if contract["protocol"]["stage"] != "development_tuning":
        raise ValueError("not a tuning batch")
    path = args.batch / "SELECTION.json"
    if path.exists():
        raise FileExistsError("refusing to replace the retained parameter selection")
    result = {
        **selection(contract["protocol"], load_records(args.batch / "results.jsonl")),
        "results_sha256": sha(args.batch / "results.jsonl"),
        "source_sha256": contract["source_sha256"],
        "source": contract["source"],
        "protocol_sha256": contract["protocol_sha256"],
        "audit": checked,
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Paired classical development tuning",
        "",
        "Sixty seconds per search, matched starts; all trials retained.",
        "",
        "| Trial | Mean at 1s | Mean at 10s | Mean at 60s |",
        "|---|---:|---:|---:|",
    ]
    for trial in result["trials"]:
        values = trial["mean_edges"]
        lines.append(
            f"| {trial['id']} | {values['1']:.3f} | {values['10']:.3f} | {values['60']:.3f} |"
        )
    lines += ["", f"Selected: {result['winners']}.", "", result["scope"], ""]
    (args.batch / "TUNING_REPORT.md").write_text("\n".join(lines))
    print(f"Selected retained development parameters: {path}")


def load_tuning(path: Path) -> SearchConfig:
    result = json.loads(path.read_text())
    batch = path.parent
    checked = independent_audit(batch)
    contract = verify_contract(batch)
    if contract["protocol"]["stage"] != "development_tuning":
        raise ValueError("not a retained development tuning selection")
    if any(result[key] != contract[key] for key in ["source", "source_sha256", "protocol_sha256"]):
        raise ValueError("tuning selection provenance differs from its contract")
    if checked["failed_records"] or result["results_sha256"] != sha(batch / "results.jsonl"):
        raise ValueError("tuning results changed or contain failures")
    regenerated = selection(contract["protocol"], load_records(batch / "results.jsonl"))
    if any(
        result[key] != regenerated[key] for key in ["selected_configuration", "winners", "trials"]
    ):
        raise ValueError("selected configuration differs from predeclared development rule")
    for filename in [
        "graph.py",
        "engine.py",
        "regions.py",
        "constructions.py",
        "search.py",
        "controls.py",
        "development.py",
    ]:
        relative = f"src/extremal_graph/repair/{filename}"
        if contract["source"].get(relative) != source_manifest().get(relative):
            raise ValueError("classical machinery changed since tuning; repeat tuning")
    config = result["selected_configuration"]
    return SearchConfig(**{**config, "region_sizes": tuple(config["region_sizes"])})

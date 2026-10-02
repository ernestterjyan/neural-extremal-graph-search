"""Stage runner for the learned C4 repair campaign (development first)."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from extremal_graph.repair.constructions import starting_pool
from extremal_graph.repair.engine import repair
from extremal_graph.repair.evidence import (
    append_record,
    json_sha,
    load_records,
    prepare_batch,
    require_development,
    source_manifest,
    verify_contract,
)
from extremal_graph.repair.graph import Graph, verify_witness
from extremal_graph.repair.regions import candidate_regions
from extremal_graph.repair.search import SearchConfig, neighborhood_search, tabu_search

EXACT = {35: 106, 40: 127}


def profile(args) -> None:
    require_development(args.sizes)
    protocol = {
        "stage": "development_profile",
        "sizes": args.sizes,
        "seed_offset": args.seed,
        "region_sizes": args.region_sizes,
        "per_family": 1,
        "repair_seconds": args.seconds,
    }
    contract = prepare_batch(args.output, protocol)
    path = args.output / "results.jsonl"
    for n in args.sizes:
        started = time.perf_counter()
        pool = starting_pool(n, args.seed + n)
        strongest = max(pool, key=lambda s: s.graph.m)
        regions = candidate_regions(
            strongest.graph,
            random.Random(args.seed + 10000 + n),
            sizes=tuple(args.region_sizes),
            per_family=1,
        )
        records = []
        for j, region in enumerate(regions):
            result = repair(
                strongest.graph, region.vertices, seconds=args.seconds, seed=args.seed + j
            )
            records.append(
                {
                    "region": asdict(region),
                    "gain": result.gain,
                    "seconds": result.seconds,
                    "status": result.status,
                    "local_optimal": result.local_optimal,
                    "rounds": result.rounds,
                    "cuts": result.cuts,
                    "edges": result.graph.edges(),
                }
            )
            print(n, region.family, len(region.vertices), result.gain, result.status, flush=True)
        record = {
            "n": n,
            "seed": args.seed + n,
            "initial_edges": strongest.graph.edges(),
            "initial_kind": strongest.kind,
            "initial_edge_count": strongest.graph.m,
            "pool": [
                {"kind": s.kind, "field_order": s.field_order, "edges": s.graph.m} for s in pool
            ],
            "repairs": records,
            "elapsed_seconds": time.perf_counter() - started,
            "source_sha256": contract["source_sha256"],
            "protocol_sha256": contract["protocol_sha256"],
        }
        append_record(path, record)


def prepare(args) -> None:
    protocol = {
        "stage": "feasibility",
        "states_per_size": 50,
        "sizes": [35, 40],
        "known_exact_values": EXACT,
        "seed_offset": args.seed,
        "state_rule": "best of identical generated start pool; skip exact optima, log exclusions",
        "region_sizes": args.region_sizes,
        "per_family": args.per_family,
        "repair_seconds": args.seconds,
        "decision": "pass iff >=10 of 100 states have verified positive gain",
        "repair_rule": "each candidate independently repairs the original state; run entire pool",
        "failure_rule": "failed states count as not improved; preserve exception records",
        "paid_compute_eur": 0.0,
    }
    prepare_batch(args.output, protocol)
    print(f"Prepared immutable feasibility contract: {args.output}")


def feasibility(args) -> None:
    contract = verify_contract(args.batch)
    protocol = contract["protocol"]
    if (
        protocol["stage"] != "feasibility"
        or json_sha(source_manifest()) != contract["source_sha256"]
    ):
        raise ValueError("run requires the frozen feasibility source and protocol")
    path = args.batch / "results.jsonl"
    done = {(r["n"], r["replication"]) for r in load_records(path)} if path.exists() else set()
    for n in protocol["sizes"]:
        for replication in range(protocol["states_per_size"]):
            if (n, replication) in done:
                continue
            started = time.perf_counter()
            record = {
                "n": n,
                "replication": replication,
                "source_sha256": contract["source_sha256"],
                "protocol_sha256": contract["protocol_sha256"],
                "repairs": [],
                "exclusions": [],
            }
            try:
                seed = protocol["seed_offset"] + n * 10000 + replication
                while True:
                    pool = starting_pool(n, seed)
                    strongest = max(pool, key=lambda s: s.graph.m)
                    verify_witness(n, strongest.graph.edges())
                    if strongest.graph.m < int(protocol["known_exact_values"][str(n)]):
                        break
                    record["exclusions"].append({"seed": seed, "edges": strongest.graph.m})
                    seed += 1000
                    if len(record["exclusions"]) > 100:
                        raise RuntimeError("failed to draw a state below the known optimum")
                record.update(
                    seed=seed,
                    initial_edges=strongest.graph.edges(),
                    initial_kind=strongest.kind,
                    initial_edge_count=strongest.graph.m,
                    starting_sha256=strongest.graph.digest(),
                    pool=[
                        {"kind": s.kind, "field_order": s.field_order, "edges": s.graph.edges()}
                        for s in pool
                    ],
                )
                regions = candidate_regions(
                    strongest.graph,
                    random.Random(seed + 100000000),
                    sizes=tuple(protocol["region_sizes"]),
                    per_family=protocol["per_family"],
                )
                for j, region in enumerate(regions):
                    outcome = repair(
                        strongest.graph,
                        region.vertices,
                        seconds=protocol["repair_seconds"],
                        seed=seed + 1000000 + j,
                    )
                    verify_witness(n, outcome.graph.edges())
                    record["repairs"].append(
                        {
                            "region": asdict(region),
                            "seed": seed + 1000000 + j,
                            "gain": outcome.gain,
                            "seconds": outcome.seconds,
                            "status": outcome.status,
                            "local_optimal": outcome.local_optimal,
                            "rounds": outcome.rounds,
                            "cuts": outcome.cuts,
                            "edges": outcome.graph.edges(),
                        }
                    )
                record["improved"] = any(r["gain"] > 0 for r in record["repairs"])
                record["status"] = "complete"
            except Exception:
                record["status"] = "failed"
                record["improved"] = False
                record["error"] = traceback.format_exc()
            record["elapsed_seconds"] = time.perf_counter() - started
            append_record(path, record)
            print(
                f"n={n} state={replication} improved={record['improved']} {record['status']}",
                flush=True,
            )


def baseline(args) -> None:
    require_development(args.sizes)
    config = SearchConfig(
        seconds=args.seconds,
        region_sizes=tuple(args.region_sizes),
        repair_seconds=args.repair_seconds,
        per_family=args.per_family,
    )
    protocol = {
        "stage": "development_baseline",
        "sizes": args.sizes,
        "methods": args.methods,
        "replications": args.replications,
        "seed_offset": args.seed,
        "configuration": asdict(config),
    }
    contract = prepare_batch(args.output, protocol)
    for n in args.sizes:
        for replication in range(args.replications):
            seed = args.seed + 10000 * n + replication
            for method in args.methods:
                try:
                    result = (
                        tabu_search(n, seed, config)
                        if method == "tabu"
                        else neighborhood_search(n, seed, config, method=method)
                    )
                    result.update(
                        replication=replication,
                        status="complete",
                        source_sha256=contract["source_sha256"],
                        protocol_sha256=contract["protocol_sha256"],
                    )
                except Exception:
                    result = {
                        "n": n,
                        "seed": seed,
                        "method": method,
                        "replication": replication,
                        "status": "failed",
                        "error": traceback.format_exc(),
                        "source_sha256": contract["source_sha256"],
                        "protocol_sha256": contract["protocol_sha256"],
                    }
                append_record(args.output / "results.jsonl", result)
                print(
                    n, method, replication, result.get("edge_count"), result["status"], flush=True
                )


def verify(args) -> list[dict]:
    contract = verify_contract(args.batch)
    records = load_records(args.batch / "results.jsonl")
    protocol = contract["protocol"]
    keys = []
    for record in records:
        if (
            record["source_sha256"] != contract["source_sha256"]
            or record["protocol_sha256"] != contract["protocol_sha256"]
        ):
            raise ValueError("result contract mismatch")
        n = record["n"]
        keys.append((n, record.get("replication", 0), record.get("method", "repair")))
        if "initial_edges" in record:
            verify_witness(n, record["initial_edges"])
            if len(record["initial_edges"]) != record["initial_edge_count"]:
                raise ValueError("initial count mismatch")
            if protocol["stage"] == "feasibility" and len(record["initial_edges"]) >= int(
                protocol["known_exact_values"][str(n)]
            ):
                raise ValueError("gate state not below optimum")
        outcomes = record.get("repairs", [])
        for outcome in outcomes if isinstance(outcomes, list) else []:
            verify_witness(n, outcome["edges"])
            initial = Graph.from_edges(n, record["initial_edges"])
            repaired = Graph.from_edges(n, outcome["edges"])
            if outcome["gain"] != repaired.m - initial.m:
                raise ValueError("repair gain mismatch")
            outside = set(range(n)) - set(outcome["region"]["vertices"])
            if any(
                initial.has(u, v) != repaired.has(u, v) for u in outside for v in outside if u < v
            ):
                raise ValueError("fixed complement changed")
        if "edges" in record:
            verify_witness(n, record["edges"])
            if len(record["edges"]) != record["edge_count"]:
                raise ValueError("count mismatch")
            for event in record["history"]:
                verify_witness(n, event["graph_edges"])
                if (
                    event["seconds"] > record["budget_seconds"]
                    or len(event["graph_edges"]) != event["edges"]
                ):
                    raise ValueError("history outside budget or count mismatch")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate experimental cells")
    if protocol["stage"] == "feasibility":
        expected = {
            (n, i, "repair") for n in protocol["sizes"] for i in range(protocol["states_per_size"])
        }
        if set(keys) != expected:
            raise ValueError(f"incomplete feasibility panel: {len(keys)}/{len(expected)} states")
    elif protocol["stage"] == "development_baseline":
        expected = {
            (n, i, method)
            for n in protocol["sizes"]
            for i in range(protocol["replications"])
            for method in protocol["methods"]
        }
        if set(keys) != expected:
            raise ValueError("incomplete baseline panel")
    elif protocol["stage"] == "development_profile" and {r["n"] for r in records} != set(
        protocol["sizes"]
    ):
        raise ValueError("incomplete development profile")
    print(f"Verified {len(records)} records and all retained witnesses.")
    return records


def report(args) -> None:
    records = verify(args)
    protocol = verify_contract(args.batch)["protocol"]
    lines = [f"# {protocol['stage']} report", "", f"Records: {len(records)}.", ""]
    if protocol["stage"] == "feasibility":
        improved = sum(r["improved"] for r in records)
        failed = sum(r["status"] == "failed" for r in records)
        passed = improved >= 10
        summary = {
            "states": len(records),
            "improved": improved,
            "failed": failed,
            "passed": passed,
            "local_cpu_seconds": sum(r["elapsed_seconds"] for r in records),
            "paid_eur": 0.0,
        }
        (args.batch / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines += [
            f"Verified improvements: **{improved}/100**. Failed states: {failed}.",
            "",
            f"Feasibility gate: **{'PASS' if passed else 'FAIL'}** (threshold 10/100).",
            "",
            "This tests the proposed repair pool on development states. It does not establish",
            "a learned advantage, held-out transfer, a new bound, or global optimality.",
            "",
            f"Local experiment time: {summary['local_cpu_seconds'] / 3600:.3f} hours.",
            "Paid compute: €0.",
        ]
    (args.batch / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ["profile", "prepare", "baseline"]:
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=2700000)
        command.add_argument("--region-sizes", type=int, nargs="+", default=[3, 5, 7])
        command.add_argument("--seconds", type=float, default=1.0 if name != "baseline" else 60.0)
        command.add_argument("--per-family", type=int, default=2)
        if name != "prepare":
            command.add_argument("--sizes", type=int, nargs="+", default=[35, 40])
        if name == "baseline":
            command.add_argument("--replications", type=int, default=10)
            command.add_argument("--repair-seconds", type=float, default=1.0)
            command.add_argument(
                "--methods",
                nargs="+",
                choices=["tabu", "adaptive", "random"],
                default=["tabu", "adaptive"],
            )
    for name in ["feasibility", "verify", "report"]:
        command = commands.add_parser(name)
        command.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    {
        "profile": profile,
        "prepare": prepare,
        "feasibility": feasibility,
        "baseline": baseline,
        "verify": verify,
        "report": report,
    }[args.stage](args)


if __name__ == "__main__":
    main()

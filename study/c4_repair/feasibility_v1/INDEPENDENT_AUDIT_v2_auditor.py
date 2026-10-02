"""Independent audit of C4 repair evidence, without importing the solver package.

Uses only Python's standard library. An audit does not independently certify a
CP-SAT optimality proof: it certifies witness feasibility and checks claim/provenance
consistency. Partial audits never authorize a gate or inferential result.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

TRAIN = {20, 24, 28, 32, 36, 40}
VALIDATION = {22, 26, 30, 34, 38}
EVALUATION = {44, 48, 52, 56, 64, 72, 80, 96}
METHODS = {"gnn", "mlp", "tabu", "adaptive"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value, label: str, *, lower: float = 0, upper: float = float("inf")) -> float:
    ensure(type(value) in (int, float) and math.isfinite(value), f"invalid {label}")
    ensure(lower <= value <= upper, f"out-of-range {label}")
    return value


def witness(n: int, raw) -> frozenset[tuple[int, int]]:
    ensure(type(n) is int and n >= 0, "invalid graph order")
    ensure(isinstance(raw, list), "witness must be an edge list")
    rows = [set() for _ in range(n)]
    pairs = set()
    for edge in raw:
        ensure(isinstance(edge, list) and len(edge) == 2, "invalid edge shape")
        u, v = edge
        ensure(type(u) is int and type(v) is int and 0 <= u < v < n, "invalid edge endpoints")
        ensure((u, v) not in pairs, "duplicate witness edge")
        pairs.add((u, v))
        rows[u].add(v)
        rows[v].add(u)
    for u, v in itertools.combinations(range(n), 2):
        ensure(len(rows[u].intersection(rows[v])) <= 1, "four-cycle in witness")
    return frozenset(pairs)


def graph_digest(n: int, graph) -> str:
    return hashlib.sha256(json.dumps([n, sorted(graph)]).encode()).hexdigest()


def contract(directory: Path) -> dict:
    data = json.loads((directory / "contract.json").read_text())
    ensure(
        canonical_digest(data["protocol"]) == data["protocol_sha256"], "protocol digest mismatch"
    )
    ensure(
        canonical_digest(data["source"]) == data["source_sha256"], "source manifest digest mismatch"
    )
    base = (directory / "source").resolve()
    for name, expected in data["source"].items():
        path = (base / name).resolve()
        ensure(path.is_relative_to(base), "source path escapes snapshot")
        ensure(digest(path) == expected, f"source snapshot changed: {name}")
    return data


def expected_cells(protocol: dict) -> set:
    stage = protocol["stage"]
    sizes = protocol["sizes"]
    ensure(len(sizes) == len(set(sizes)), "duplicate protocol sizes")
    if stage == "feasibility":
        ensure(
            set(sizes) == {35, 40} and protocol["states_per_size"] == 50,
            "feasibility requires 50 states at each of n35/n40",
        )
    elif stage == "collection":
        ensure(protocol["split"] in {"train", "validation"}, "unknown collection split")
        ensure(
            set(sizes) == (TRAIN if protocol["split"] == "train" else VALIDATION),
            "collection size/split leakage",
        )
    elif stage in {"validation", "evaluation"}:
        ensure(
            set(sizes) == (VALIDATION if stage == "validation" else EVALUATION),
            "wrong controlled size panel",
        )
        ensure(
            protocol["blocks"] == 8 and protocol["replications"] == 10,
            "controlled panel must preserve eight blocks and ten starting seeds",
        )
        ensure(
            set(protocol["methods"]) == METHODS and len(protocol["methods"]) == 4,
            "missing or duplicate designated method",
        )
        ensure(protocol["configuration"]["seconds"] == 60, "primary search budget changed")
        return {(n, i, m, b) for n in sizes for i in range(10) for m in METHODS for b in range(8)}
    elif stage == "development_baseline":
        ensure(all(2 <= n <= 40 for n in sizes), "held-out baseline leakage")
        return {
            (n, i, m, 0)
            for n in sizes
            for i in range(protocol["replications"])
            for m in protocol["methods"]
        }
    elif stage == "development_profile":
        ensure(all(2 <= n <= 40 for n in sizes), "held-out profile leakage")
        return {(n, 0, "repair", 0) for n in sizes}
    else:
        raise ValueError(f"unsupported witness batch stage: {stage}")
    return {(n, i, "repair", 0) for n in sizes for i in range(protocol["states_per_size"])}


def audit(directory: Path, *, allow_partial: bool = False) -> dict:
    metadata = contract(directory)
    protocol = metadata["protocol"]
    stage = protocol["stage"]
    expected = expected_cells(protocol)
    raw = (directory / "results.jsonl").read_bytes()
    complete_bytes = raw if not raw or raw.endswith(b"\n") else raw[: raw.rfind(b"\n") + 1]
    partial_bytes = len(raw) - len(complete_bytes)
    ensure(allow_partial or partial_bytes == 0, "unfinished evidence line")
    records = [json.loads(line) for line in complete_bytes.splitlines() if line.strip()]
    found = set()
    verified = 0
    failures = []
    improved = Counter()
    timing_records = legacy_timing_records = 0
    local_certificates = 0
    starts = {}
    controlled = stage in {"validation", "evaluation"}
    runner = directory / "source/experiments/c4_repair.py"
    event_schema = runner.exists() and '"incumbent_events"' in runner.read_text()

    for record in records:
        n = record["n"]
        block, replication = record.get("block", 0), record.get("replication", 0)
        method = record.get("method", "repair")
        key = (n, replication, method, block)
        ensure(key in expected and key not in found, "unexpected or duplicate experimental cell")
        found.add(key)
        ensure(record["source_sha256"] == metadata["source_sha256"], "record source mismatch")
        ensure(record["protocol_sha256"] == metadata["protocol_sha256"], "record protocol mismatch")
        ensure(record.get("status", "complete") in {"complete", "failed"}, "unterminated cell")
        if record.get("status") == "failed":
            ensure(bool(record.get("error")), "failure has no retained error")
            failures.append(key)
            if stage == "feasibility":
                ensure(record["improved"] is False, "failed state credited as an improvement")
        elif stage != "development_profile":
            seed = protocol["seed_offset"] + n * 10000 + replication
            if controlled:
                seed += block * 1000000
            elif stage == "feasibility":
                seed += 1000 * len(record["exclusions"])
            ensure(record["seed"] == seed, "record seed differs from its declared schedule")
        else:
            ensure(record["seed"] == protocol["seed_offset"] + n, "profile seed mismatch")

        initial = None
        if "initial_edges" in record:
            initial = witness(n, record["initial_edges"])
            verified += 1
            ensure(len(initial) == record["initial_edge_count"], "initial edge-count mismatch")
            if "starting_sha256" in record:
                ensure(
                    graph_digest(n, initial) == record["starting_sha256"],
                    "initial witness hash mismatch",
                )
            if stage == "feasibility":
                ensure(
                    len(initial) < int(protocol["known_exact_values"][str(n)]),
                    "gate state at optimum",
                )

        outcomes = record.get("repairs", [])
        if isinstance(outcomes, list):
            unique_regions = set()
            positive = False
            for outcome in outcomes:
                ensure(initial is not None, "repair has no starting witness")
                repaired = witness(n, outcome["edges"])
                verified += 1
                ensure(len(repaired) - len(initial) == outcome["gain"] >= 0, "repair gain mismatch")
                region = outcome["region"]["vertices"]
                ensure(
                    len(region) == len(set(region))
                    and all(type(u) is int and 0 <= u < n for u in region),
                    "invalid repair region",
                )
                ensure(tuple(sorted(region)) not in unique_regions, "duplicate labelled region")
                unique_regions.add(tuple(sorted(region)))
                selected = set(region)
                fixed_initial = {e for e in initial if not selected.intersection(e)}
                fixed_final = {e for e in repaired if not selected.intersection(e)}
                ensure(fixed_initial == fixed_final, "repair changed fixed complement")
                if outcome["local_optimal"]:
                    ensure(
                        outcome["status"] == "OPTIMAL",
                        "local certificate with nonoptimal solver status",
                    )
                    local_certificates += 1
                if event_schema:
                    ensure("incumbent_events" in outcome, "new repair record lacks event timings")
                if "incumbent_events" in outcome:
                    best, previous = len(initial), -1.0
                    for event in outcome["incumbent_events"]:
                        elapsed = finite(
                            event["seconds"], "incumbent time", upper=protocol["repair_seconds"]
                        )
                        ensure(elapsed >= previous, "incumbent time reversed")
                        graph = witness(n, event["graph_edges"])
                        verified += 1
                        ensure(len(graph) == event["edges"] > best, "nonimproving incumbent event")
                        previous, best = elapsed, len(graph)
                    ensure(best == len(repaired), "repair gain has no within-budget event")
                    timing_records += 1
                else:
                    legacy_timing_records += 1
                positive |= outcome["gain"] > 0
            if stage == "feasibility" and record["status"] == "complete":
                ensure(positive == record["improved"], "state improvement flag mismatch")
                improved[n] += positive

        for start in record.get("starting_pool", record.get("pool", [])):
            if isinstance(start.get("edges"), list):
                graph = witness(n, start["edges"])
                verified += 1
                if "sha256" in start:
                    ensure(
                        graph_digest(n, graph) == start["sha256"], "starting pool digest mismatch"
                    )
        if stage == "feasibility" and record["status"] == "complete":
            pool_graphs = [frozenset(map(tuple, s["edges"])) for s in record["pool"]]
            ensure(
                bool(pool_graphs) and initial in pool_graphs, "gate state not in its starting pool"
            )
            ensure(len(initial) == max(map(len, pool_graphs)), "gate state is weaker than its pool")
        if controlled and record["status"] == "complete":
            start_key = (n, replication, block)
            pool = record["starting_pool"]
            ensure(
                start_key not in starts or starts[start_key] == pool,
                "unequal comparator starting pools",
            )
            starts[start_key] = pool
            if method in {"gnn", "mlp"}:
                model = next(
                    m for m in protocol["models"] if m["block"] == block and m["family"] == method
                )
                ensure(
                    record["checkpoint_sha256"] == model["checkpoint_sha256"]
                    and record["training_seed"] == model["seed"],
                    "wrong trained model for block",
                )

        if "edges" in record:
            final = witness(n, record["edges"])
            verified += 1
            ensure(len(final) == record["edge_count"], "final edge-count mismatch")
            budget = finite(record["budget_seconds"], "search budget", lower=1e-9)
            if controlled:
                ensure(budget == 60, "controlled search changed budget")
            best, previous = -1, -1.0
            for event in record["history"]:
                elapsed = finite(event["seconds"], "history time", upper=budget)
                ensure(elapsed >= previous, "history time reversed")
                graph = witness(n, event["graph_edges"])
                verified += 1
                ensure(len(graph) == event["edges"] > best, "invalid improvement history")
                best, previous = len(graph), elapsed
            ensure(best == len(final), "final score differs from the best credited history")
            ensure(
                frozenset(map(tuple, record["history"][-1]["graph_edges"])) == final,
                "final witness differs from credited incumbent",
            )
            checkpoints = {str(t) for t in [1, 10, 60] if t <= budget}
            ensure(set(record["checkpoints"]) == checkpoints, "missing or extra timed checkpoints")
            for cutoff, event in record["checkpoints"].items():
                expected_event = max(
                    (e for e in record["history"] if e["seconds"] <= int(cutoff)),
                    key=lambda e: e["edges"],
                )
                ensure(event == expected_event, "checkpoint differs from available history")

    complete = found == expected and not partial_bytes
    ensure(allow_partial or complete, f"incomplete panel: {len(found)}/{len(expected)}")
    return {
        "stage": stage,
        "complete_panel": complete,
        "records": len(found),
        "expected_records": len(expected),
        "unfinished_line_bytes": partial_bytes,
        "failed_records": failures,
        "verified_graphs": verified,
        "timed_repair_records": timing_records,
        "legacy_timing_records": legacy_timing_records,
        "claimed_local_certificates_checked": local_certificates,
        "local_certificates_independently_proved": False,
        "improved_states_by_size": dict(improved),
        "feasibility_passed": stage == "feasibility" and complete and sum(improved.values()) >= 10,
        "eligible_for_inference": controlled and complete and not failures,
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "results_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "auditor_sha256": digest(Path(__file__)),
        "limitations": [
            "Set-based witness checks, not a separate CP optimality proof checker.",
            "Stored monotonic timestamps are checked for consistency, not externally remeasured.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.batch, allow_partial=args.allow_partial)
    if args.output:
        ensure(not args.output.exists(), "refusing to overwrite an audit")
        auditor_snapshot = args.output.with_name(args.output.stem + "_auditor.py")
        ensure(not auditor_snapshot.exists(), "refusing to overwrite an auditor snapshot")
        auditor_snapshot.write_bytes(Path(__file__).read_bytes())
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

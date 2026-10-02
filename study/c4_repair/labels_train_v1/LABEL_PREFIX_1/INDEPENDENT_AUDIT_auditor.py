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
import random
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
    elif stage == "reproduction":
        parent = protocol["reference_protocol"]
        ensure(
            canonical_digest(parent) == protocol["reference_protocol_sha256"],
            "reproduction reference protocol digest mismatch",
        )
        parent_stage = protocol["reference_stage"]
        ensure(parent_stage in {"validation", "evaluation"}, "invalid reproduction reference")
        declared_sizes = [22, 30, 38] if parent_stage == "validation" else [44, 64, 96]
        declaration = {
            "block": 0,
            "replication": 0,
            "sizes": declared_sizes,
            "methods": ["gnn", "mlp", "adaptive", "tabu"],
        }
        ensure(
            parent["stage"] == parent_stage
            and parent.get("reproduction_subset") == declaration
            and sizes == declared_sizes
            and protocol["block"] == protocol["replication"] == 0,
            "reproduction differs from prospectively declared subset",
        )
        expected_cells(parent)  # Parent remains the full controlled contract.
        for name in ["configuration", "models", "methods", "seed_offset", "method_order"]:
            ensure(protocol[name] == parent[name], f"reproduction changed {name}")
        return {(n, 0, m, 0) for n in sizes for m in METHODS}
    elif stage == "development_tuning":
        ensure(all(2 <= n <= 40 for n in sizes), "held-out tuning leakage")
        trials = protocol["trials"]
        identifiers = [t["id"] for t in trials]
        ensure(len(identifiers) == len(set(identifiers)), "duplicate tuning trial")
        ensure(
            {t["method"] for t in trials} == {"tabu", "adaptive"},
            "both classical families required",
        )
        ensure(all(t["configuration"]["seconds"] == 60 for t in trials), "tuning budget changed")
        return {
            (n, i, trial, 0)
            for n in sizes
            for i in range(protocol["replications"])
            for trial in identifiers
        }
    elif stage in {
        "development_profile",
        "development_selector_profile",
        "development_trained_selector_profile",
    }:
        ensure(all(2 <= n <= 40 for n in sizes), "held-out profile leakage")
        if stage == "development_trained_selector_profile":
            ensure(set(sizes) == {20, 35, 40}, "matched profile size panel changed")
            ensure(
                protocol["repeats"] == 20
                and protocol["warmups"] == 1
                and protocol["amortization_decisions_per_search"] == 30,
                "matched profile scenario changed",
            )
            models = protocol["models"]
            ensure(
                len(models) == 16
                and {(m["family"], m["block"]) for m in models}
                == {(f, b) for f in ["gnn", "mlp"] for b in range(8)},
                "matched profile needs all sixteen models",
            )
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
    unlogged_equal_score_replacements = 0
    local_certificates = 0
    starts = {}
    controlled = stage in {"validation", "evaluation"}
    reproduction = stage == "reproduction"
    if reproduction:
        ensure(
            metadata["source_sha256"] == protocol["reference_source_sha256"],
            "reproduction changed reference source",
        )
        ensure(
            metadata["environment"]["python_prefix"]
            != protocol["reference_environment"]["python_prefix"],
            "reproduction did not use a distinct Python environment",
        )
    runner = directory / "source/experiments/c4_repair.py"
    event_schema = runner.exists() and '"incumbent_events"' in runner.read_text()
    timed_search = (
        controlled or reproduction or stage in {"development_baseline", "development_tuning"}
    )

    for record in records:
        n = record["n"]
        block, replication = record.get("block", 0), record.get("replication", 0)
        method = record.get("method", "repair")
        key = (n, replication, record.get("trial", method), block)
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
        elif stage not in {
            "development_profile",
            "development_selector_profile",
            "development_trained_selector_profile",
        }:
            seed = protocol["seed_offset"] + n * 10000 + replication
            if controlled or reproduction:
                seed += block * 1000000
            elif stage == "feasibility":
                seed += 1000 * len(record["exclusions"])
            ensure(record["seed"] == seed, "record seed differs from its declared schedule")
        else:
            ensure(record["seed"] == protocol["seed_offset"] + n, "profile seed mismatch")

        if record.get("status", "complete") == "complete":
            if timed_search:
                configuration = protocol.get("configuration")
                if stage == "development_tuning":
                    trial = next(t for t in protocol["trials"] if t["id"] == record["trial"])
                    ensure(method == trial["method"], "tuning trial method mismatch")
                    configuration = trial["configuration"]
                    order = list(protocol["trials"])
                    random.Random(record["seed"] + 500000000).shuffle(order)
                    ensure(
                        type(record["trial_position"]) is int
                        and 0 <= record["trial_position"] < len(order),
                        "invalid tuning position",
                    )
                    ensure(
                        order[record["trial_position"]]["id"] == record["trial"],
                        "wrong tuning order",
                    )
                required = {
                    "edges",
                    "edge_count",
                    "history",
                    "checkpoints",
                    "budget_seconds",
                    "elapsed_seconds",
                    "configuration",
                    "starting_pool",
                    "initialization_seconds",
                }
                ensure(required <= record.keys(), "completed search lacks required evidence")
                ensure(
                    record["configuration"] == configuration,
                    "search configuration differs from protocol",
                )
                ensure(
                    record["budget_seconds"] == configuration["seconds"],
                    "search budget differs from protocol",
                )
                finite(record["elapsed_seconds"], "search elapsed time")
                finite(record["initialization_seconds"], "initialization time")
            else:
                ensure(
                    {"initial_edges", "initial_edge_count", "repairs"} <= record.keys(),
                    "completed repair state lacks required evidence",
                )
                ensure(isinstance(record["repairs"], list), "invalid repair outcome panel")
            if stage == "development_selector_profile":
                ensure(not record["repairs"], "selector profiling cannot claim repair outcomes")
                ensure(
                    {m["family"] for m in record["models"]} == {"gnn", "mlp"},
                    "missing profile family",
                )
                for model in record["models"]:
                    ensure(model["untrained"] is True, "profile model not labelled untrained")
                    ensure(
                        len(model["scores"]) == len(record["regions"]), "wrong profile score count"
                    )
                    ensure(
                        len(model["inference_seconds"]) == protocol["repeats"], "missing timings"
                    )
                    for score in model["scores"]:
                        ensure(math.isfinite(score), "nonfinite profile score")
                    for duration in model["inference_seconds"]:
                        finite(duration, "inference time")
            if stage == "development_trained_selector_profile":
                ensure(not record["repairs"], "selection latency profile claims repair results")
                order = [*protocol["models"], {"family": "adaptive", "block": None}]
                random.Random(record["seed"] + 200000000).shuffle(order)
                conditions = record["conditions"]
                ensure(len(conditions) == 17, "missing matched selection condition")
                for position, condition in enumerate(conditions):
                    ensure(condition["position"] == position, "wrong profile condition order")
                    ensure(
                        all(condition[key] == value for key, value in order[position].items()),
                        "wrong profile checkpoint or condition",
                    )
                    ensure(
                        len(condition["selection_seconds"])
                        == len(condition["selected_indices"])
                        == 20,
                        "incomplete matched selection timings",
                    )
                    for duration in condition["selection_seconds"]:
                        finite(duration, "selection time")
                    for index in condition["selected_indices"]:
                        ensure(
                            type(index) is int and 0 <= index < len(record["regions"]),
                            "profile selected an unavailable candidate",
                        )
                    if condition["family"] != "adaptive":
                        scores = condition["last_scores"]
                        ensure(len(scores) == len(record["regions"]), "wrong trained score count")
                        ensure(all(math.isfinite(s) for s in scores), "nonfinite trained score")
                        ensure(
                            scores[condition["selected_indices"][-1]] == max(scores),
                            "trained selector did not choose a highest-ranked candidate",
                        )

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
                        ensure(
                            {e for e in graph if not selected.intersection(e)} == fixed_initial,
                            "repair incumbent changed fixed complement",
                        )
                        previous, best = elapsed, len(graph)
                    ensure(best == len(repaired), "repair gain has no within-budget event")
                    credited = (
                        frozenset(map(tuple, outcome["incumbent_events"][-1]["graph_edges"]))
                        if outcome["incumbent_events"]
                        else initial
                    )
                    # Frozen v2 accepts feasible equal-score replacements before
                    # its deadline but records only strict improvements. The timed
                    # witness proves the score; independently feasible equal-score
                    # finals are disclosed rather than silently assigned that time.
                    unlogged_equal_score_replacements += credited != repaired
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
        if timed_search and record["status"] == "complete":
            start_key = (n, replication, block)
            pool = record["starting_pool"]
            ensure(
                start_key not in starts or starts[start_key] == pool,
                "unequal comparator starting pools",
            )
            starts[start_key] = pool
            if "method_order" in protocol:
                order = sorted(protocol["methods"])
                random.Random(protocol["seed_offset"] + n * 10000 + replication).shuffle(order)
                offset = block % len(order)
                order = order[offset:] + order[:offset]
                ensure(
                    type(record["method_position"]) is int
                    and 0 <= record["method_position"] < len(order),
                    "invalid paired method position",
                )
                ensure(order[record["method_position"]] == method, "wrong paired method order")
            if (controlled or reproduction) and method in {"gnn", "mlp"}:
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
            if controlled or reproduction:
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
        "repair_equal_score_replacements_without_event": unlogged_equal_score_replacements,
        "claimed_local_certificates_checked": local_certificates,
        "local_certificates_independently_proved": False,
        "improved_states_by_size": dict(improved),
        "feasibility_passed": stage == "feasibility" and complete and sum(improved.values()) >= 10,
        "eligible_for_inference": controlled and complete and not failures,
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "results_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "results_bytes": len(complete_bytes),
        "auditor_sha256": digest(Path(__file__)),
        "limitations": [
            "Set-based witness checks, not a separate CP optimality proof checker.",
            "Stored monotonic timestamps are checked for consistency, not externally remeasured.",
            "Repair equal-score replacement witnesses may lack an event; only the recorded "
            "improvement witness receives its timestamp.",
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

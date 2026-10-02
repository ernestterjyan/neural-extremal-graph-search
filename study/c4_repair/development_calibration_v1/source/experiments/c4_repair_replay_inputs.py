"""Replay generated starts and complete label pools from an audited source snapshot.

No repair optimization or model training is performed. Graph feasibility is audited
separately by c4_repair_audit; this checks deterministic input generation/completeness.
For timed search, only the complete initial starting pool is replayed, not the search
trajectory, repair outcomes or elapsed-time performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

AUDITOR_PATH = Path(__file__).with_name("c4_repair_audit.py")
if Path(__file__).stem.endswith("_replay"):
    AUDITOR_PATH = Path(__file__).with_name(
        Path(__file__).stem.removesuffix("_replay") + "_auditor.py"
    )
auditor_spec = importlib.util.spec_from_file_location("input_replay_auditor", AUDITOR_PATH)
auditor = importlib.util.module_from_spec(auditor_spec)
auditor_spec.loader.exec_module(auditor)
audit, digest, ensure = auditor.audit, auditor.digest, auditor.ensure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    ensure(not args.output.exists(), "refusing to overwrite input replay evidence")
    independent = audit(args.batch, allow_partial=args.allow_partial)
    metadata = json.loads((args.batch / "contract.json").read_text())
    protocol = metadata["protocol"]
    stage = protocol["stage"]
    timed = stage in {"development_baseline", "development_tuning", "validation", "evaluation"}
    ensure(
        timed
        or stage
        in {"feasibility", "collection", "development_profile", "development_selector_profile"},
        "unsupported input replay stage",
    )
    source = (args.batch / "source/src").resolve()
    sys.path.insert(0, str(source))
    constructions = importlib.import_module("extremal_graph.repair.constructions")
    regions_module = importlib.import_module("extremal_graph.repair.regions")
    graph_module = importlib.import_module("extremal_graph.repair.graph")
    for module in [constructions, regions_module, graph_module]:
        ensure(
            Path(module.__file__).resolve().is_relative_to(source),
            "loaded live rather than frozen source",
        )
    results = []
    # A live batch can append while we audit. Replay exactly the audited complete
    # prefix, ignoring later appends and an unfinished final line.
    complete_bytes = (args.batch / "results.jsonl").read_bytes()[: independent["results_bytes"]]
    ensure(
        hashlib.sha256(complete_bytes).hexdigest() == independent["results_sha256"],
        "audited evidence prefix changed before replay",
    )
    records = [json.loads(line) for line in complete_bytes.splitlines() if line.strip()]
    for record in records:
        identity = {
            "n": record["n"],
            "replication": record.get("replication", 0),
            "block": record.get("block", 0),
            "method": record.get("method", "repair"),
        }
        if "trial" in record:
            identity["trial"] = record["trial"]
        if record.get("status") == "failed":
            results.append(
                {
                    **identity,
                    "status": "retained_failure",
                    "input_replayed": False,
                }
            )
            continue
        n, seed = record["n"], record["seed"]
        pool = constructions.starting_pool(n, seed)
        best = max(pool, key=lambda s: s.graph.m)
        initial = best.graph.copy()
        if timed:
            expected_pool = [
                {
                    "kind": s.kind,
                    "field_order": s.field_order,
                    "sha256": s.graph.digest(),
                    "edges": [list(e) for e in s.graph.edges()],
                }
                for s in pool
            ]
            ensure(expected_pool == record["starting_pool"], "timed starting pool does not replay")
            results.append(
                {
                    **identity,
                    "status": "replayed",
                    "input_replayed": True,
                    "input_scope": "complete initial starting pool only",
                    "starting_sha256": initial.digest(),
                    "starting_graphs": len(pool),
                }
            )
            continue
        if stage == "collection":
            state_type = record["replication"] % 5
            ensure(record["state_type"] == state_type, "collection state-type mismatch")
            if state_type in {1, 2, 3}:
                initial = pool[{1: 0, 2: 1, 3: -1}[state_type]].graph.copy()
            elif state_type == 4:
                rng = random.Random(seed + 50000000)
                selected = set(rng.sample(range(n), 3))
                for edge in initial.edges():
                    if set(edge).intersection(selected):
                        initial.remove(*edge)
                graph_module.greedy_fill(initial, rng)
        elif stage == "feasibility":
            expected_pool = [
                {
                    "kind": s.kind,
                    "field_order": s.field_order,
                    "edges": [list(e) for e in s.graph.edges()],
                }
                for s in pool
            ]
            ensure(expected_pool == record["pool"], "starting pool does not replay")
        elif stage == "development_profile":
            expected_pool = [
                {"kind": s.kind, "field_order": s.field_order, "edges": s.graph.m} for s in pool
            ]
            ensure(expected_pool == record["pool"], "profile pool does not replay")
        ensure(
            [list(e) for e in initial.edges()] == record["initial_edges"],
            "starting witness does not replay",
        )
        candidate_seed = (
            protocol["seed_offset"] + 10000 + n
            if stage == "development_profile"
            else seed + 100000000
        )
        regions = regions_module.candidate_regions(
            initial,
            random.Random(candidate_seed),
            sizes=tuple(protocol["region_sizes"]),
            per_family=protocol["per_family"],
        )
        if stage == "development_selector_profile":
            recorded = [(tuple(r["vertices"]), r["family"]) for r in record["regions"]]
        else:
            recorded = [
                (tuple(r["region"]["vertices"]), r["region"]["family"]) for r in record["repairs"]
            ]
        expected = [(r.vertices, r.family) for r in regions]
        ensure(
            recorded == expected, "candidate pool is incomplete or differs from seeded generation"
        )
        if stage in {"feasibility", "collection"}:
            ensure(
                all(r["seed"] == seed + 1000000 + j for j, r in enumerate(record["repairs"])),
                "repair seed schedule changed",
            )
        results.append(
            {
                **identity,
                "status": "replayed",
                "input_replayed": True,
                "starting_sha256": initial.digest(),
                "candidate_regions": len(regions),
            }
        )
    report = {
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "results_sha256": independent["results_sha256"],
        "results_bytes": independent["results_bytes"],
        "replay_program_sha256": digest(Path(__file__)),
        "independent_auditor_sha256": independent["auditor_sha256"],
        "complete_panel": independent["complete_panel"],
        "cells": results,
        "all_completed_inputs_replayed": all(
            r["input_replayed"] for r in results if r["status"] != "retained_failure"
        ),
        "repair_outcomes_rerun": False,
        "timed_search_trajectories_rerun": False,
        "timed_performance_reproduced": False,
        "scope": (
            "complete initial starting pools only; no timed search or repair trajectory replay"
            if timed
            else "complete generated repair states and ordered candidate pools"
        ),
    }
    snapshot = args.output.with_name(args.output.stem + "_replay.py")
    ensure(not snapshot.exists(), "refusing to replace replay source")
    auditor_snapshot = args.output.with_name(args.output.stem + "_auditor.py")
    ensure(not auditor_snapshot.exists(), "refusing to replace replay auditor source")
    ensure(
        digest(AUDITOR_PATH) == independent["auditor_sha256"],
        "auditor source changed during replay",
    )
    snapshot.write_bytes(Path(__file__).read_bytes())
    auditor_snapshot.write_bytes(AUDITOR_PATH.read_bytes())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Replayed {sum(r['input_replayed'] for r in results)} complete input states.")


if __name__ == "__main__":
    main()

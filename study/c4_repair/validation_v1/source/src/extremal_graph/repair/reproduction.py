"""Predeclared timed subset reproduction in a distinct locked Python environment."""

from __future__ import annotations

import json
import time
import traceback

import torch

from .campaign import checkpoint_manifest
from .controls import balanced_method_order
from .evidence import (
    REPRODUCTION_SIZES,
    append_record,
    environment,
    independent_audit,
    json_sha,
    load_records,
    prepare_batch,
    prepare_or_resume,
    require_input_replay,
    sha,
    source_manifest,
    verify_contract,
)
from .graph import verify_witness
from .search import SearchConfig, neighborhood_search, tabu_search
from .selectors import model_selector
from .training import load_model


def declared_subset(protocol: dict) -> dict:
    stage = protocol["stage"]
    if stage not in REPRODUCTION_SIZES:
        raise ValueError("only completed controlled validation/evaluation may be reproduced")
    expected = {
        "block": 0,
        "replication": 0,
        "sizes": list(REPRODUCTION_SIZES[stage]),
        "methods": ["gnn", "mlp", "adaptive", "tabu"],
    }
    if protocol.get("reproduction_subset") != expected:
        raise ValueError("reference lacks the prospectively declared reproduction subset")
    return expected


def prepare(args) -> None:
    reference = verify_contract(args.reference)
    checked = independent_audit(args.reference)
    if not checked["eligible_for_inference"]:
        raise ValueError("complete, nonfailed independently audited reference required")
    require_input_replay(args.reference)
    parent = reference["protocol"]
    subset = declared_subset(parent)
    if json_sha(source_manifest()) != reference["source_sha256"]:
        raise ValueError("use the exact archived reference source for reproduction")
    models = checkpoint_manifest(args.checkpoints)
    if models != parent["models"]:
        raise ValueError("reproduction requires the original trained checkpoint panel")
    original_prefix = reference["environment"].get("python_prefix")
    current = environment()
    if original_prefix is None or current["python_prefix"] == original_prefix:
        raise ValueError("reproduction requires a distinct Python environment")
    if not current["hardware"]["verified_cpu_model"]:
        raise ValueError("record verified reproduction hardware identity")
    for key in ["python", "packages", "cpu_workers"]:
        if current[key] != reference["environment"][key]:
            raise ValueError(f"reproduction must preserve the locked runtime: {key}")
    protocol = {
        "stage": "reproduction",
        "sizes": subset["sizes"],
        "methods": subset["methods"],
        "block": subset["block"],
        "replication": subset["replication"],
        "seed_offset": parent["seed_offset"],
        "configuration": parent["configuration"],
        "models": models,
        "method_order": parent["method_order"],
        "reference_stage": parent["stage"],
        "reference_protocol": parent,
        "reference_protocol_sha256": reference["protocol_sha256"],
        "reference_source_sha256": reference["source_sha256"],
        "reference_contract_sha256": sha(args.reference / "contract.json"),
        "reference_results_sha256": checked["results_sha256"],
        "reference_environment": reference["environment"],
        "scope": "fresh environment, fixed subset; no new gate or primary inferential result",
    }
    prepare_batch(args.output, protocol)
    print(f"Frozen 12 timed reproduction searches: {args.output}")


def run(args) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = verify_contract(args.batch)
    protocol = metadata["protocol"]
    if protocol["stage"] != "reproduction":
        raise ValueError("not a reproduction batch")
    declared_subset(protocol["reference_protocol"])
    prepare_or_resume(args.batch, protocol)
    if checkpoint_manifest(args.checkpoints) != protocol["models"]:
        raise ValueError("trained model panel changed before reproduction")
    config = SearchConfig(
        **{
            **protocol["configuration"],
            "region_sizes": tuple(protocol["configuration"]["region_sizes"]),
        }
    )
    path = args.batch / "results.jsonl"
    existing = load_records(path) if path.exists() else []
    done = {(r["n"], r["method"]) for r in existing}
    if len(done) != len(existing):
        raise ValueError("duplicate reproduction cell")
    if existing:
        independent_audit(args.batch, allow_partial=True)
    for n in protocol["sizes"]:
        seed = protocol["seed_offset"] + n * 10000
        order = balanced_method_order(
            protocol["methods"], seed_offset=protocol["seed_offset"], n=n, replication=0, block=0
        )
        for position, method in enumerate(order):
            if (n, method) in done:
                continue
            record = {
                "n": n,
                "block": 0,
                "replication": 0,
                "seed": seed,
                "method": method,
                "method_position": position,
                "status": "complete",
                "source_sha256": metadata["source_sha256"],
                "protocol_sha256": metadata["protocol_sha256"],
            }
            try:
                selector = None
                started = time.perf_counter()
                if method in {"gnn", "mlp"}:
                    model = next(
                        m for m in protocol["models"] if m["block"] == 0 and m["family"] == method
                    )
                    checkpoint = args.checkpoints / method / str(model["seed"]) / "best.pt"
                    selector = model_selector(load_model(checkpoint, model["checkpoint_sha256"]))
                    record.update(
                        checkpoint_sha256=model["checkpoint_sha256"], training_seed=model["seed"]
                    )
                record["model_loading_seconds"] = time.perf_counter() - started
                result = (
                    tabu_search(n, seed, config, started_at=started)
                    if method == "tabu"
                    else neighborhood_search(
                        n, seed, config, method=method, selector=selector, started_at=started
                    )
                )
                record.update(result)
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
            append_record(path, record)
            print(n, method, record.get("edge_count"), record["status"], flush=True)


def report(args) -> None:
    checked = independent_audit(args.batch)
    if checked["failed_records"]:
        raise ValueError("retain reproduction failures; do not omit them from comparison")
    require_input_replay(args.batch)
    contract = verify_contract(args.batch)
    protocol = contract["protocol"]
    if (
        sha(args.reference / "results.jsonl") != protocol["reference_results_sha256"]
        or sha(args.reference / "contract.json") != protocol["reference_contract_sha256"]
    ):
        raise ValueError("reference evidence changed")
    reference = {
        (r["n"], r["method"]): r
        for r in load_records(args.reference / "results.jsonl")
        if r["block"] == 0 and r["replication"] == 0
    }
    differences = []
    for row in load_records(args.batch / "results.jsonl"):
        parent = reference[(row["n"], row["method"])]
        if row["starting_pool"] != parent["starting_pool"]:
            raise ValueError("reproduction starting pool differs from reference")
        verify_witness(row["n"], row["edges"], explicit_cycles=True)
        differences.append(
            {
                "n": row["n"],
                "method": row["method"],
                "reference_edges": parent["edge_count"],
                "reproduced_edges": row["edge_count"],
                "edge_difference": row["edge_count"] - parent["edge_count"],
                "same_final_witness": row["edges"] == parent["edges"],
                "checkpoint_edge_differences": {
                    t: row["checkpoints"][t]["edges"] - parent["checkpoints"][t]["edges"]
                    for t in ["1", "10", "60"]
                },
            }
        )
    result = {
        "audit": checked,
        "cells": differences,
        "distinct_python_environment": contract["environment"]["python_prefix"]
        != protocol["reference_environment"]["python_prefix"],
        "same_hardware_identity": contract["environment"]["hardware"]
        == protocol["reference_environment"]["hardware"],
        "separate_environment_is_not_independent_hardware": True,
        "all_finals_explicitly_c4_verified": True,
        "all_inputs_equal_reference": True,
        "scope": protocol["scope"],
    }
    output = args.batch / "REPRODUCTION_REPORT.json"
    if output.exists():
        raise FileExistsError("refusing to replace reproduction report")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))

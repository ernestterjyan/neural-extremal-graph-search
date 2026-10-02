"""Synthetic contract fixtures test reproduction scope, not timed performance."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from extremal_graph.repair import reproduction
from extremal_graph.repair.controls import balanced_method_order
from extremal_graph.repair.evidence import (
    append_record,
    environment,
    independent_audit,
    json_sha,
    prepare_batch,
    source_manifest,
)
from extremal_graph.repair.search import SearchConfig


def protocol():
    parent = {
        "stage": "validation",
        "sizes": [22, 26, 30, 34, 38],
        "blocks": 8,
        "replications": 10,
        "methods": ["gnn", "mlp", "adaptive", "tabu"],
        "configuration": asdict(SearchConfig()),
        "seed_offset": 7000000,
        "method_order": "seeded permutation per size/replication; rotate across blocks",
        "models": [
            {"family": f, "block": b, "seed": 6000000 + b, "checkpoint_sha256": "a" * 64}
            for f in ["gnn", "mlp"]
            for b in range(8)
        ],
        "reproduction_subset": {
            "block": 0,
            "replication": 0,
            "sizes": [22, 30, 38],
            "methods": ["gnn", "mlp", "adaptive", "tabu"],
        },
    }
    return {
        **{
            k: parent[k]
            for k in ["methods", "configuration", "models", "seed_offset", "method_order"]
        },
        "stage": "reproduction",
        "sizes": [22, 30, 38],
        "block": 0,
        "replication": 0,
        "reference_stage": "validation",
        "reference_protocol": parent,
        "reference_protocol_sha256": json_sha(parent),
        "reference_source_sha256": json_sha(source_manifest()),
        "reference_environment": {**environment(), "python_prefix": "/reference-fixture"},
    }


def empty_search(config):
    event = {"seconds": 0, "edges": 0, "graph_edges": []}
    return {
        "edges": [],
        "edge_count": 0,
        "history": [event],
        "checkpoints": {str(t): event for t in [1, 10, 60]},
        "budget_seconds": 60,
        "elapsed_seconds": 60,
        "configuration": asdict(config),
        "starting_pool": [],
        "initialization_seconds": 0,
    }


def synthetic_batch(tmp_path):
    batch = tmp_path / "reproduction"
    p = protocol()
    metadata = prepare_batch(batch, p)
    for n in p["sizes"]:
        order = balanced_method_order(
            p["methods"], seed_offset=7000000, n=n, replication=0, block=0
        )
        for position, method in enumerate(order):
            row = {
                **empty_search(SearchConfig()),
                "n": n,
                "block": 0,
                "replication": 0,
                "seed": 7000000 + n * 10000,
                "method": method,
                "method_position": position,
                "status": "complete",
                "source_sha256": metadata["source_sha256"],
                "protocol_sha256": metadata["protocol_sha256"],
            }
            if method in {"gnn", "mlp"}:
                row.update(checkpoint_sha256="a" * 64, training_seed=6000000)
            append_record(batch / "results.jsonl", row)
    return batch


def test_declared_reproduction_is_fixed_before_outcomes():
    parent = protocol()["reference_protocol"]
    assert reproduction.declared_subset(parent)["sizes"] == [22, 30, 38]
    parent["reproduction_subset"]["sizes"] = [22, 26, 30]
    with pytest.raises(ValueError, match="prospectively declared"):
        reproduction.declared_subset(parent)
    with pytest.raises(ValueError, match="only completed controlled"):
        reproduction.declared_subset({"stage": "development_baseline"})


def test_reproduction_audit_does_not_authorize_primary_inference(tmp_path):
    batch = synthetic_batch(tmp_path)
    checked = independent_audit(batch)
    assert checked["complete_panel"] and checked["records"] == 12
    assert not checked["eligible_for_inference"] and not checked["feasibility_passed"]


@pytest.mark.parametrize("changed", ["same_environment", "budget", "source"])
def test_reproduction_audit_rejects_scope_and_provenance_changes(tmp_path, changed):
    batch = synthetic_batch(tmp_path)
    path = batch / "contract.json"
    metadata = json.loads(path.read_text())
    p = metadata["protocol"]
    if changed == "same_environment":
        metadata["environment"]["python_prefix"] = p["reference_environment"]["python_prefix"]
    elif changed == "budget":
        p["configuration"]["seconds"] = 30
    else:
        p["reference_source_sha256"] = "b" * 64
    metadata["protocol_sha256"] = json_sha(p)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="distinct Python|changed configuration|reference source"):
        independent_audit(batch)


def test_reproduction_charges_loading_uses_original_cells_and_retains_resume(tmp_path, monkeypatch):
    batch = tmp_path / "repeat"
    p = protocol()
    prepare_batch(batch, p)
    calls, loaded = [], []
    monkeypatch.setattr(reproduction, "checkpoint_manifest", lambda directory: p["models"])

    def load(path, checksum):
        loaded.append((path, checksum))
        return "synthetic model fixture"

    def search(n, seed, config, **kwargs):
        assert kwargs["started_at"] <= reproduction.time.perf_counter()
        calls.append((n, seed, kwargs.get("method", "tabu")))
        return empty_search(config)

    monkeypatch.setattr(reproduction, "load_model", load)
    monkeypatch.setattr(reproduction, "model_selector", lambda model: model)
    monkeypatch.setattr(reproduction, "tabu_search", search)
    monkeypatch.setattr(reproduction, "neighborhood_search", search)
    args = SimpleNamespace(batch=batch, checkpoints=tmp_path / "synthetic_models")
    reproduction.run(args)
    assert len(calls) == 12 and len(loaded) == 6
    assert all(seed == 7000000 + n * 10000 for n, seed, _ in calls)
    path = batch / "results.jsonl"
    retained = path.read_bytes()
    reproduction.run(args)
    assert len(calls) == 12 and path.read_bytes() == retained
    assert independent_audit(batch)["complete_panel"]

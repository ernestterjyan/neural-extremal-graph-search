"""Frozen calibration can resume without replacing completed or failed cells."""

import json
from types import SimpleNamespace

import pytest

from experiments import c4_repair
from extremal_graph.repair.evidence import independent_audit, load_records


def prepared(tmp_path):
    args = SimpleNamespace(
        output=tmp_path / "baseline",
        sizes=[6],
        methods=["construction", "simple_repair"],
        replications=2,
        seed=123,
        configuration=None,
        seconds=0.03,
        region_sizes=[2],
        repair_seconds=0.01,
        per_family=1,
    )
    c4_repair.prepare_baseline(args)
    return args.output


def test_resume_keeps_completed_prefix_and_finishes_same_panel(tmp_path):
    batch = prepared(tmp_path)
    args = SimpleNamespace(batch=batch)
    c4_repair.run_baseline(args)
    path = batch / "results.jsonl"
    prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:3])
    path.write_bytes(prefix)
    c4_repair.run_baseline(args)
    assert path.read_bytes().startswith(prefix)
    assert independent_audit(batch)["complete_panel"]
    assert len(load_records(path)) == 4
    complete = path.read_bytes()
    c4_repair.run_baseline(args)
    assert path.read_bytes() == complete


def test_failed_cells_are_retained_without_hidden_retry(tmp_path, monkeypatch):
    batch = prepared(tmp_path)
    calls = []

    def failed(*args, **kwargs):
        calls.append(kwargs["method"])
        raise RuntimeError("intentional failure fixture")

    monkeypatch.setattr(c4_repair, "calibration_search", failed)
    args = SimpleNamespace(batch=batch)
    c4_repair.run_baseline(args)
    c4_repair.run_baseline(args)
    rows = load_records(batch / "results.jsonl")
    assert len(calls) == len(rows) == 4
    assert all(r["status"] == "failed" and "intentional failure" in r["error"] for r in rows)
    assert len(independent_audit(batch)["failed_records"]) == 4


def test_resume_refuses_changed_runtime(tmp_path):
    batch = prepared(tmp_path)
    path = batch / "contract.json"
    contract = json.loads(path.read_text())
    contract["environment"]["packages"]["torch"] = "different"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="environment change"):
        c4_repair.run_baseline(SimpleNamespace(batch=batch))
    assert not (batch / "results.jsonl").exists()


def test_resume_refuses_duplicate_cells(tmp_path):
    batch = prepared(tmp_path)
    c4_repair.run_baseline(SimpleNamespace(batch=batch))
    path = batch / "results.jsonl"
    with path.open("ab") as stream:
        stream.write(path.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        c4_repair.run_baseline(SimpleNamespace(batch=batch))

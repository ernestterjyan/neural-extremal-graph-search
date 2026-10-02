"""Incomplete and failed evidence stays visible without authorizing inference."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/c4_repair_state_report.py"
spec = importlib.util.spec_from_file_location("repair_state_report", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def batch(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    protocol = {
        "stage": "feasibility",
        "sizes": [35, 40],
        "states_per_size": 50,
        "seed_offset": 100,
        "repair_seconds": 2,
        "known_exact_values": {"35": 106, "40": 127},
    }
    metadata = {
        "protocol": protocol,
        "protocol_sha256": module.auditor.canonical_digest(protocol),
        "source": {},
        "source_sha256": module.auditor.canonical_digest({}),
    }
    (directory / "contract.json").write_text(json.dumps(metadata))
    record = {
        "n": 35,
        "replication": 0,
        "seed": 350100,
        "exclusions": [],
        "status": "complete",
        "initial_edges": [],
        "initial_edge_count": 0,
        "pool": [{"edges": []}],
        "repairs": [],
        "improved": False,
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
    }
    return directory, record


def test_incomplete_gate_is_pending_even_with_valid_witnesses(tmp_path):
    directory, record = batch(tmp_path)
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
    report = module.build_report(directory)
    assert report["feasibility"]["gate_status"] == "PENDING"
    assert not report["inferential_result_available"]
    assert "1/100" in module.markdown(report)


def test_failed_state_error_is_visible_and_not_credited(tmp_path):
    directory, record = batch(tmp_path)
    record.update(status="failed", error="Backend failed before returning a witness.")
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
    report = module.build_report(directory)
    assert report["failed_cells"] == 1 and report["completed_cells"] == 0
    assert report["feasibility"]["improved_states"] == 0
    assert record["error"] in module.markdown(report)
    assert not report["inferential_result_available"]


def test_corrupt_witness_generates_an_explicit_invalid_report(tmp_path):
    directory, record = batch(tmp_path)
    record["initial_edges"] = [[0, 1], [0, 3], [1, 2], [2, 3]]
    record["initial_edge_count"] = 4
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
    report = module.build_report(directory)
    assert report["audit_status"] == "invalid_or_unavailable"
    assert "four-cycle" in report["error"]
    assert not report["inferential_result_available"]
    assert "No result is authorized" in module.markdown(report)


def test_retained_report_runs_with_its_own_auditor(tmp_path):
    directory, record = batch(tmp_path)
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
    output = tmp_path / "c4_repair_state.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--batch", str(directory), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    second = tmp_path / "AGAIN.json"
    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "c4_repair_state_report.py"),
            "--batch",
            str(directory),
            "--output",
            str(second),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(second.read_text())["feasibility"]
        == json.loads(output.read_text())["feasibility"]
    )

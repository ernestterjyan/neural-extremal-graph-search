"""Evidence corruption checks for the separate standard-library auditor."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/c4_repair_audit.py"
spec = importlib.util.spec_from_file_location("repair_auditor", SCRIPT)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def make_profile(tmp_path):
    protocol = {
        "stage": "development_profile",
        "sizes": [4],
        "seed_offset": 10,
        "repair_seconds": 1,
    }
    directory = tmp_path / "batch"
    directory.mkdir()
    metadata = {
        "protocol": protocol,
        "protocol_sha256": auditor.canonical_digest(protocol),
        "source": {},
        "source_sha256": auditor.canonical_digest({}),
    }
    (directory / "contract.json").write_text(json.dumps(metadata))
    record = {
        "n": 4,
        "seed": 14,
        "initial_edge_count": 1,
        "initial_edges": [[0, 1]],
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "repairs": [
            {
                "region": {"vertices": [1], "family": "random"},
                "gain": 1,
                "edges": [[0, 1], [1, 2]],
                "local_optimal": False,
                "status": "FEASIBLE",
                "incumbent_events": [{"seconds": 0.5, "edges": 2, "graph_edges": [[0, 1], [1, 2]]}],
            }
        ],
    }
    return directory, record


def write(directory, record):
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")


def test_independent_common_neighbor_checker():
    auditor.witness(4, [[0, 1], [1, 2], [2, 3]])
    with pytest.raises(ValueError, match="four-cycle"):
        auditor.witness(4, [[0, 1], [1, 2], [2, 3], [0, 3]])
    with pytest.raises(ValueError, match="duplicate"):
        auditor.witness(4, [[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="endpoints"):
        auditor.witness(4, [[False, 1]])


def test_complete_valid_repair_evidence(tmp_path):
    directory, record = make_profile(tmp_path)
    write(directory, record)
    report = auditor.audit(directory)
    assert report["complete_panel"] and report["verified_graphs"] == 3
    assert report["timed_repair_records"] == 1
    assert not report["eligible_for_inference"]


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("late", "incumbent time"),
        ("uncounted", "within-budget"),
        ("complement", "fixed complement"),
        ("provenance", "source mismatch"),
        ("invalid", "four-cycle"),
        ("duplicate", "duplicate experimental"),
    ],
)
def test_false_results_cannot_pass_audit(tmp_path, mutation, message):
    directory, record = make_profile(tmp_path)
    outcome = record["repairs"][0]
    if mutation == "late":
        outcome["incumbent_events"][0]["seconds"] = 1.01
    elif mutation == "uncounted":
        outcome["incumbent_events"] = []
    elif mutation == "complement":
        outcome["edges"] = [[0, 1], [2, 3]]
        outcome["incumbent_events"][0]["graph_edges"] = outcome["edges"]
    elif mutation == "provenance":
        record["source_sha256"] = "invented"
    elif mutation == "invalid":
        outcome["edges"] = [[0, 1], [1, 2], [2, 3], [0, 3]]
        outcome["gain"] = 3
    write(directory, record)
    if mutation == "duplicate":
        with (directory / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=message):
        auditor.audit(directory)


def test_partial_audit_does_not_assert_completion(tmp_path):
    directory, record = make_profile(tmp_path)
    (directory / "results.jsonl").write_text(json.dumps(record)[:30])
    report = auditor.audit(directory, allow_partial=True)
    assert not report["complete_panel"] and report["records"] == 0
    assert report["unfinished_line_bytes"] == 30
    with pytest.raises(ValueError, match="unfinished"):
        auditor.audit(directory)


def test_controlled_panel_cannot_omit_comparator_or_replication():
    protocol = {
        "stage": "evaluation",
        "sizes": list(auditor.EVALUATION),
        "blocks": 8,
        "replications": 10,
        "methods": ["gnn", "mlp", "adaptive", "tabu"],
        "configuration": {"seconds": 60},
    }
    assert len(auditor.expected_cells(protocol)) == 2560
    protocol["methods"].remove("mlp")
    with pytest.raises(ValueError, match="method"):
        auditor.expected_cells(protocol)
    protocol["methods"].append("mlp")
    protocol["replications"] = 9
    with pytest.raises(ValueError, match="ten starting"):
        auditor.expected_cells(protocol)


def test_source_snapshot_tampering_is_detected(tmp_path):
    directory, record = make_profile(tmp_path)
    source = directory / "source/code.py"
    source.parent.mkdir()
    source.write_text("original")
    metadata = json.loads((directory / "contract.json").read_text())
    metadata["source"] = {"code.py": hashlib.sha256(b"original").hexdigest()}
    metadata["source_sha256"] = auditor.canonical_digest(metadata["source"])
    (directory / "contract.json").write_text(json.dumps(metadata))
    record["source_sha256"] = metadata["source_sha256"]
    write(directory, record)
    source.write_text("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        auditor.audit(directory)


def test_repair_event_must_preserve_complement(tmp_path):
    directory, record = make_profile(tmp_path)
    event = record["repairs"][0]["incumbent_events"][0]
    event["graph_edges"] = [[0, 1], [2, 3]]
    write(directory, record)
    with pytest.raises(ValueError, match="fixed complement"):
        auditor.audit(directory)


def test_feasible_equal_score_replacement_is_disclosed(tmp_path):
    directory, record = make_profile(tmp_path)
    record["repairs"][0]["edges"] = [[0, 1], [1, 3]]
    write(directory, record)
    report = auditor.audit(directory)
    assert report["repair_equal_score_replacements_without_event"] == 1
    assert report["timed_repair_records"] == 1


@pytest.mark.parametrize("omitted", ["edges", "history", "starting_pool"])
def test_completed_search_cannot_omit_its_evidence(tmp_path, omitted):
    directory, _ = make_profile(tmp_path)
    metadata = json.loads((directory / "contract.json").read_text())
    protocol = {
        "stage": "development_baseline",
        "sizes": [4],
        "replications": 1,
        "methods": ["tabu"],
        "seed_offset": 10,
        "configuration": {"seconds": 1},
    }
    metadata.update(protocol=protocol, protocol_sha256=auditor.canonical_digest(protocol))
    (directory / "contract.json").write_text(json.dumps(metadata))
    event = {"seconds": 0, "edges": 0, "graph_edges": []}
    record = {
        "n": 4,
        "seed": 40010,
        "method": "tabu",
        "source_sha256": metadata["source_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "status": "complete",
        "edges": [],
        "edge_count": 0,
        "history": [event],
        "checkpoints": {"1": event},
        "budget_seconds": 1,
        "elapsed_seconds": 1,
        "configuration": {"seconds": 1},
        "starting_pool": [],
        "initialization_seconds": 0,
    }
    write(directory, record)
    assert auditor.audit(directory)["complete_panel"]
    del record[omitted]
    write(directory, record)
    with pytest.raises(ValueError, match="lacks required evidence"):
        auditor.audit(directory)

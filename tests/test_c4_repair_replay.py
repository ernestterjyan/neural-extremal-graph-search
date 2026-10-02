"""Input replay checks against real retained frozen development evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from extremal_graph.repair.evidence import require_input_replay

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/c4_repair_replay_inputs.py"


def run(script, batch, output, *args):
    return subprocess.run(
        [sys.executable, str(script), "--batch", str(batch), "--output", str(output), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_real_inputs_and_retained_replay_program(tmp_path):
    batch = ROOT / "study/c4_repair/profile_v3"
    output = tmp_path / "INPUTS.json"
    result = run(SCRIPT, batch, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["complete_panel"] and report["all_completed_inputs_replayed"]
    assert len(report["cells"]) == 2 and not report["repair_outcomes_rerun"]
    # The retained program also carries its standard-library auditor, so no
    # import from the current checkout is needed to reproduce its audit.
    second = tmp_path / "REPLAYED.json"
    result = run(tmp_path / "INPUTS_replay.py", batch, second)
    assert result.returncode == 0, result.stderr
    assert json.loads(second.read_text())["cells"] == report["cells"]


def test_incomplete_candidate_pool_is_rejected(tmp_path):
    batch = tmp_path / "batch"
    shutil.copytree(ROOT / "study/c4_repair/profile_v3", batch)
    rows = [json.loads(line) for line in (batch / "results.jsonl").read_text().splitlines()]
    rows[0]["repairs"].pop()
    (batch / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run(SCRIPT, batch, tmp_path / "INPUTS.json")
    assert result.returncode != 0
    assert "candidate pool is incomplete" in result.stderr


def test_partial_final_line_is_not_replayed_or_called_complete(tmp_path):
    batch = tmp_path / "batch"
    shutil.copytree(ROOT / "study/c4_repair/profile_v3", batch)
    rows = (batch / "results.jsonl").read_bytes().splitlines()
    (batch / "results.jsonl").write_bytes(rows[0] + b"\n" + rows[1][:35])
    output = tmp_path / "INPUTS.json"
    result = run(SCRIPT, batch, output, "--allow-partial")
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert len(report["cells"]) == 1 and not report["complete_panel"]
    assert report["results_bytes"] == len(rows[0]) + 1


def test_real_timed_starting_pools_replay_without_claiming_timed_performance(tmp_path):
    batch = ROOT / "study/c4_repair/baseline_smoke_v2"
    output = tmp_path / "INPUTS.json"
    result = run(SCRIPT, batch, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["complete_panel"] and len(report["cells"]) == 12
    assert report["all_completed_inputs_replayed"]
    assert not report["timed_performance_reproduced"]
    assert not report["timed_search_trajectories_rerun"]
    assert {c["method"] for c in report["cells"]} == {
        "tabu",
        "adaptive",
        "construction",
        "simple_repair",
    }
    second = tmp_path / "REPLAYED.json"
    result = run(tmp_path / "INPUTS_replay.py", batch, second)
    assert result.returncode == 0, result.stderr
    assert json.loads(second.read_text())["cells"] == report["cells"]


def test_identically_truncated_starting_pools_are_rejected(tmp_path):
    # Matching pools across methods can still all omit the same generated start.
    # Witness feasibility does not by itself establish input-generation completeness.
    batch = tmp_path / "batch"
    shutil.copytree(ROOT / "study/c4_repair/baseline_smoke_v2", batch)
    rows = [json.loads(line) for line in (batch / "results.jsonl").read_text().splitlines()]
    for row in rows:
        if row["n"] == 20:
            row["starting_pool"].pop()
    (batch / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run(SCRIPT, batch, tmp_path / "INPUTS.json")
    assert result.returncode != 0
    assert "timed starting pool does not replay" in result.stderr


@pytest.mark.parametrize("tampering", ["omitted", "duplicated", "completion_flag"])
def test_replay_guard_checks_actual_cell_coverage(tmp_path, tampering):
    batch = tmp_path / "batch"
    shutil.copytree(ROOT / "study/c4_repair/baseline_smoke_v2", batch)
    path = batch / "INPUT_REPLAY.json"
    result = run(SCRIPT, batch, path)
    assert result.returncode == 0, result.stderr
    require_input_replay(batch)
    report = json.loads(path.read_text())
    if tampering == "omitted":
        report["cells"].pop()
    elif tampering == "duplicated":
        report["cells"][0] = report["cells"][1]
    else:
        report["cells"][0]["input_replayed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="omitted or duplicated|completion differs"):
        require_input_replay(batch)


def test_unchanged_historical_gate_replay_still_satisfies_coverage_guard():
    require_input_replay(ROOT / "study/c4_repair/feasibility_v2")

"""A real two-step matched-training interruption preserves the scientific run."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import matched_exposure as study  # noqa: E402


def test_interrupted_training_replays_weights_and_rejects_changed_contract(tmp_path, monkeypatch):
    protocol = {
        "threads_per_worker": 1,
        "iterations": 2,
        "training_size": 14,
        "rollouts_per_iteration": 4,
        "elite_fraction": 0.20,
        "elite_trajectories_per_iteration": 1,
        "elite_prefix_length": 12,
        "updates_per_iteration": 1,
        "batch_size": 8,
        "validation_interval": 1,
        "validation_episodes": 2,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(study, "PROTOCOL", protocol_path)
    job = ("candidate", 9)
    monkeypatch.setattr(study, "ART", tmp_path / "uninterrupted")
    study.train_cell(job)
    baseline = torch.load(
        tmp_path / "uninterrupted/matched_training/candidate-seed-9/best.pt",
        map_location="cpu",
        weights_only=False,
    )

    monkeypatch.setattr(study, "ART", tmp_path / "interrupted")
    original = study._collect_rollouts

    def interrupted(*args, **kwargs):
        if kwargs["global_step"] == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "_collect_rollouts", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.train_cell(job)
    folder = tmp_path / "interrupted/matched_training/candidate-seed-9"
    metadata_before = (folder / "metadata.json").read_bytes()
    protocol_path.write_text(json.dumps(protocol | {"batch_size": 9}))
    with pytest.raises(ValueError, match="resume contract mismatch"):
        study.train_cell(job)
    assert (folder / "metadata.json").read_bytes() == metadata_before

    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(study, "_collect_rollouts", original)
    study.train_cell(job)
    resumed = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    assert baseline["iteration"] == resumed["iteration"]
    assert baseline["validation_score"] == resumed["validation_score"]
    assert all(
        torch.equal(baseline["model_state"][key], resumed["model_state"][key])
        for key in baseline["model_state"]
    )
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert sum(row["optimizer_updates"] for row in records) == 2

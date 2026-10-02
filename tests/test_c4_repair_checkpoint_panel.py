"""Untrained binary fixtures test model/seed binding, never research performance."""

import json
from pathlib import Path

import pytest
import torch

from extremal_graph.repair.campaign import checkpoint_manifest
from extremal_graph.repair.evidence import append_record, json_sha, load_records, prepare_batch, sha
from extremal_graph.repair.selectors import make_model


def panel(tmp_path: Path):
    folder = tmp_path / "panel"
    seeds = list(range(8))
    prepare_batch(
        folder,
        {
            "stage": "training",
            "training_seeds": seeds,
            "data_sha256": "synthetic-unit-fixture",
            "validation_data_sha256": "synthetic-unit-fixture",
        },
    )
    for family in ["gnn", "mlp"]:
        for seed in seeds:
            model = make_model(family)
            path = folder / family / str(seed) / "best.pt"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    "family": family,
                    "seed": seed,
                    "architecture": model.architecture,
                    "state_dict": model.state_dict(),
                },
                path,
            )
            append_record(
                folder / "models.jsonl",
                {
                    "family": family,
                    "seed": seed,
                    "architecture": model.architecture,
                    "checkpoint_sha256": sha(path),
                    "elapsed_seconds": 0,
                    "train_results_sha256": "synthetic-unit-fixture",
                    "validation_results_sha256": "synthetic-unit-fixture",
                    "fixture_untrained": True,
                },
            )
    return folder


def test_distinct_seed_and_family_headers_are_bound_to_manifest(tmp_path):
    rows = checkpoint_manifest(panel(tmp_path))
    assert len(rows) == 16
    assert {(r["family"], r["block"]) for r in rows} == {
        (family, block) for family in ["gnn", "mlp"] for block in range(8)
    }


@pytest.mark.parametrize("field,value", [("seed", 1), ("family", "mlp"), ("architecture", "old")])
def test_hash_matching_checkpoint_with_wrong_header_is_rejected(tmp_path, field, value):
    folder = panel(tmp_path)
    path = folder / "gnn/0/best.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    rows = load_records(folder / "models.jsonl")
    # Update the hash to establish that byte integrity alone is insufficient.
    rows[0]["checkpoint_sha256"] = sha(path)
    (folder / "models.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="training seed or architecture"):
        checkpoint_manifest(folder)


def test_eight_entries_cannot_substitute_for_eight_distinct_training_seeds(tmp_path):
    folder = panel(tmp_path)
    contract_path = folder / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["protocol"]["training_seeds"] = [0] * 8
    contract["protocol_sha256"] = json_sha(contract["protocol"])
    contract_path.write_text(json.dumps(contract))
    rows = load_records(folder / "models.jsonl")
    rows = [rows[0]] * 8 + [rows[8]] * 8
    (folder / "models.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="independently trained"):
        checkpoint_manifest(folder)

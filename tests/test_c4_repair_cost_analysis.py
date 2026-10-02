"""Cost scenarios use synthetic timings and cannot establish a solver advantage."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import c4_repair_trained_profile
from extremal_graph.repair.cost_analysis import amortization, analyze_costs
from extremal_graph.repair.evidence import independent_audit, require_input_replay
from extremal_graph.repair.search import SearchConfig
from extremal_graph.repair.selectors import RegionGNN, RegionMLP, make_model


def test_inference_only_recovery_is_conditional_and_never_finite_without_saving():
    positive = amortization(0.6, 0.9, 0.01)
    assert positive["searches_to_recover_model_training_only"] == 2
    assert positive["searches_to_recover_training_labels_and_tuning"] == 5
    assert "conditional" in positive["interpretation"]
    for saving in [0, -0.01]:
        negative = amortization(0.6, 0.9, saving)
        assert not negative["finite_inference_only_break_even"]
        assert negative["searches_to_recover_training_labels_and_tuning"] is None
    with pytest.raises(ValueError, match="nonnegative offline"):
        amortization(-1, 0, 0.1)
    with pytest.raises(ValueError, match="finite latency"):
        amortization(1, 0, float("nan"))


def costs_fixture():
    models = [
        {"family": f, "seed": i, "elapsed_seconds": 1} for f in ["gnn", "mlp"] for i in range(8)
    ]
    ledger = [
        {"stage": stage, "local_wall_seconds": seconds, "paid_eur": 0}
        for stage, seconds in [
            ("labels_train_fixture", 1),
            ("labels_validation_fixture", 2),
            ("tuning_fixture", 4),
            ("development_calibration_fixture", 7),
            ("models_fixture", 16),
        ]
    ]
    profile = [
        {
            "n": n,
            "conditions": [
                {"family": "adaptive", "selection_seconds": [0.001] * 20},
                *[
                    {
                        "family": m["family"],
                        "seed": m["seed"],
                        "selection_seconds": [0.004 if m["family"] == "gnn" else 0.0005] * 20,
                    }
                    for m in models
                ],
            ],
        }
        for n in [20, 35, 40]
    ]
    return models, ledger, profile


def test_costs_remain_separate_and_shared_cost_is_not_added_per_model():
    result = analyze_costs(*costs_fixture())
    assert result["local_seconds_recorded"] == 30
    assert result["model_training_seconds"] == 16
    assert result["label_collection_seconds"] == 3
    assert result["classical_tuning_seconds"] == 4
    assert result["calibration_seconds"] == 7
    assert result["paid_eur_recorded"] == 0
    assert not any(
        m["finite_inference_only_break_even"]
        for m in result["model_scenarios"]
        if m["family"] == "gnn"
    )
    assert all(
        m["searches_to_recover_training_labels_and_tuning"] == 534
        for m in result["model_scenarios"]
        if m["family"] == "mlp"
    )


@pytest.mark.parametrize("changed", ["duplicate", "negative", "missing_model", "missing_graph"])
def test_incomplete_or_double_counted_cost_evidence_is_rejected(changed):
    models, ledger, profile = costs_fixture()
    if changed == "duplicate":
        ledger.append(ledger[0])
    elif changed == "negative":
        ledger[0]["paid_eur"] = -1
    elif changed == "missing_model":
        models.pop()
    else:
        profile.pop()
    with pytest.raises(ValueError):
        analyze_costs(models, ledger, profile)


def test_matched_profile_audit_and_input_replay_with_untrained_test_fixtures(tmp_path, monkeypatch):
    # Mock checkpoint loading uses untrained fixtures only. This exercises the
    # production profiler/auditor contract, not trained research evidence.
    models = [
        {
            "family": f,
            "seed": i,
            "block": i,
            "checkpoint_sha256": "a" * 64,
            "architecture": RegionGNN.architecture if f == "gnn" else RegionMLP.architecture,
            "training_seconds": 1,
        }
        for f in ["gnn", "mlp"]
        for i in range(8)
    ]
    recipe = {"region_sizes": [3, 4, 5], "per_family": 4, "repair_seconds": 2.0}
    monkeypatch.setattr(c4_repair_trained_profile, "checkpoint_manifest", lambda path: models)
    monkeypatch.setattr(
        c4_repair_trained_profile,
        "verify_contract",
        lambda path: {"protocol": {"repair_configuration": recipe}},
    )
    monkeypatch.setattr(
        c4_repair_trained_profile,
        "load_tuning",
        lambda path: SearchConfig(region_sizes=(3, 4, 5), per_family=4, repair_seconds=2),
    )
    monkeypatch.setattr(
        c4_repair_trained_profile, "load_model", lambda path, checksum: make_model(path.parts[-3])
    )
    batch = tmp_path / "synthetic_profile"
    c4_repair_trained_profile.profile(
        SimpleNamespace(
            output=batch, checkpoints=tmp_path / "models", configuration=tmp_path / "selection"
        )
    )
    checked = independent_audit(batch)
    assert checked["complete_panel"] and checked["records"] == 3
    assert not checked["eligible_for_inference"]
    script = Path(__file__).resolve().parents[1] / "experiments/c4_repair_replay_inputs.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--batch",
            str(batch),
            "--output",
            str(batch / "INPUT_REPLAY.json"),
        ],
        check=True,
        capture_output=True,
    )
    require_input_replay(batch)
    path = batch / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["conditions"][0]["selected_indices"][0] = 100000
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="unavailable candidate"):
        independent_audit(batch)

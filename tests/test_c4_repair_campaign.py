"""Research contract checks: block replication, inference, and held-out protection."""

import math
from types import SimpleNamespace

import pytest

from extremal_graph.repair.analysis import analyze, paired_interval, student_cdf, student_quantile
from extremal_graph.repair.evidence import EVALUATION_SIZES, require_development


def panel(sizes, block_advantages):
    return [
        {
            "n": n,
            "method": method,
            "block": block,
            "replication": replication,
            "status": "complete",
            "checkpoints": {
                str(t): {"edges": 100 + (block_advantages[block] if method == "gnn" else 0)}
                for t in [1, 10, 60]
            },
        }
        for n in sizes
        for block in range(8)
        for replication in range(10)
        for method in ["gnn", "mlp", "tabu", "adaptive"]
    ]


def test_numeric_student_distribution_against_closed_forms():
    for t in [0, 0.1, 1, 3, 10]:
        assert student_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-9)
        assert student_cdf(t, 2) == pytest.approx(0.5 + t / (2 * math.sqrt(2 + t * t)), abs=1e-9)
    assert student_quantile(0.975, 7) == pytest.approx(2.364624251, abs=1e-7)
    assert student_cdf(student_quantile(0.9875, 7), 7) == pytest.approx(0.9875, abs=1e-10)


def test_primary_requires_both_comparators_and_preregistered_panel():
    records = panel(EVALUATION_SIZES, [2] * 8)
    result = analyze(records, list(EVALUATION_SIZES))
    assert result["primary_passed"]
    assert result["sizes_better_than_all_comparators"] == 8
    assert not analyze(records, list(EVALUATION_SIZES), checkpoint=10)["primary_passed"]
    for record in records:
        if record["method"] == "tabu":
            record["checkpoints"]["60"]["edges"] = 103
    assert not analyze(records, list(EVALUATION_SIZES))["primary_passed"]


def test_graph_count_does_not_inflate_model_replication():
    # Hundreds of generated graphs cannot turn eight highly variable model
    # blocks into a precisely estimated positive architecture effect.
    advantages = [-12, -10, -7, 0, 5, 9, 11, 20]
    records = panel(EVALUATION_SIZES, advantages)
    result = analyze(records, list(EVALUATION_SIZES))
    interval = paired_interval(advantages)
    assert result["comparisons"]["adaptive"]["blocks"] == advantages
    assert result["comparisons"]["adaptive"]["lower"] == interval["lower"]
    assert interval["lower"] < 0 < interval["upper"]
    assert not result["primary_passed"]


def test_incomplete_and_failed_panels_cannot_be_silently_analyzed():
    records = panel([22, 26, 30, 34, 38], [2] * 8)
    with pytest.raises(ValueError, match="complete"):
        analyze(records[:-1], [22, 26, 30, 34, 38])
    records[0]["status"] = "failed"
    with pytest.raises(ValueError, match="failed"):
        analyze(records, [22, 26, 30, 34, 38])


def test_missing_mlp_cannot_pass_and_positive_mlp_cannot_replace_primary():
    records = panel(EVALUATION_SIZES, [0] * 8)
    with pytest.raises(ValueError, match="MLP"):
        analyze([r for r in records if r["method"] != "mlp"], list(EVALUATION_SIZES))
    for record in records:
        if record["method"] == "mlp":
            record["checkpoints"]["60"]["edges"] = 103
    result = analyze(records, list(EVALUATION_SIZES))
    assert result["method_means"]["mlp"]["mean_edges"] == 103
    assert result["secondary_mlp_comparisons"]["adaptive"]["mean"] == 3
    assert not result["primary_passed"] and not result["learning_gate_passed"]


def test_held_out_orders_rejected_by_development_runner():
    require_development([20, 35, 40])
    for n in EVALUATION_SIZES:
        with pytest.raises(ValueError, match="held-out"):
            require_development([20, n])


def test_training_checkpoint_and_label_replay(tmp_path):
    from dataclasses import asdict

    import torch

    from extremal_graph.repair.engine import repair
    from extremal_graph.repair.evidence import append_record, prepare_batch
    from extremal_graph.repair.graph import Graph
    from extremal_graph.repair.regions import Region
    from extremal_graph.repair.selectors import model_selector
    from extremal_graph.repair.training import labelled_states, load_model, train_family

    torch.set_num_threads(1)
    graph = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    regions = [
        Region((0,), "random"),
        Region((1, 3), "neighborhood"),
        Region((0, 2, 4), "blocking_path"),
    ]
    labels = []
    for region in regions:
        result = repair(graph, region.vertices, seconds=2, seed=44)
        labels.append(
            {"region": asdict(region), "gain": result.gain, "edges": result.graph.edges()}
        )
    directories = []
    for split in ["train", "validation"]:
        directory = tmp_path / split
        contract = prepare_batch(
            directory, {"stage": "collection", "split": split, "sizes": [5], "states_per_size": 1}
        )
        append_record(
            directory / "results.jsonl",
            {
                "n": 5,
                "replication": 0,
                "status": "complete",
                "initial_edges": graph.edges(),
                "repairs": labels,
                "source_sha256": contract["source_sha256"],
                "protocol_sha256": contract["protocol_sha256"],
            },
        )
        directories.append(directory)
    assert len(labelled_states(directories[0])) == 1
    for family in ["gnn", "mlp"]:
        output = tmp_path / family
        metadata = train_family(*directories, output, family=family, seed=77, epochs=2)
        first = load_model(output / "best.pt", metadata["checkpoint_sha256"])
        second = load_model(output / "best.pt", metadata["checkpoint_sha256"])
        assert model_selector(first)(graph, regions) == model_selector(second)(graph, regions)
        with pytest.raises(ValueError, match="digest"):
            load_model(output / "best.pt", "wrong")


def test_no_freeze_from_incomplete_validation(tmp_path):
    from extremal_graph.repair.campaign import freeze
    from extremal_graph.repair.evidence import append_record, prepare_batch

    validation = tmp_path / "validation"
    prepare_batch(validation, {"stage": "validation"})
    append_record(validation / "results.jsonl", panel([22], [2] * 8)[0])
    with pytest.raises(ValueError, match="complete"):
        freeze(SimpleNamespace(output=tmp_path / "frozen.json", validation=validation))
    assert not (tmp_path / "frozen.json").exists()

"""Paired tuning completeness, selection policy and actual tabu tenure behavior."""

import itertools
import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from extremal_graph.repair import search
from extremal_graph.repair.development import (
    load_tuning,
    report_tuning,
    selection,
    trial_configurations,
)
from extremal_graph.repair.evidence import append_record, independent_audit, prepare_batch
from extremal_graph.repair.graph import Graph
from extremal_graph.repair.search import SearchConfig


def tuning_panel(tmp_path):
    trials = trial_configurations(
        SearchConfig(region_sizes=(3, 4, 5), per_family=4, repair_seconds=2)
    )
    protocol = {
        "stage": "development_tuning",
        "sizes": [25, 31, 35, 40],
        "replications": 3,
        "seed_offset": 10000000,
        "trials": trials,
    }
    directory = tmp_path / "tuning"
    contract = prepare_batch(directory, protocol)
    rows = []
    event = {"seconds": 0, "edges": 0, "graph_edges": []}
    for n in protocol["sizes"]:
        for replication in range(3):
            seed = protocol["seed_offset"] + n * 10000 + replication
            order = list(trials)
            random.Random(seed + 500000000).shuffle(order)
            for position, trial in enumerate(order):
                row = {
                    "n": n,
                    "seed": seed,
                    "replication": replication,
                    "method": trial["method"],
                    "trial": trial["id"],
                    "trial_position": position,
                    "status": "complete",
                    "edges": [],
                    "edge_count": 0,
                    "history": [event],
                    "checkpoints": {str(t): event for t in [1, 10, 60]},
                    "budget_seconds": 60,
                    "elapsed_seconds": 60,
                    "configuration": trial["configuration"],
                    "starting_pool": [],
                    "initialization_seconds": 0,
                    "source_sha256": contract["source_sha256"],
                    "protocol_sha256": contract["protocol_sha256"],
                }
                append_record(directory / "results.jsonl", row)
                rows.append(row)
    return directory, protocol, rows


def test_complete_tuning_panel_and_predeclared_tie_break(tmp_path):
    directory, protocol, rows = tuning_panel(tmp_path)
    audit = independent_audit(directory)
    assert audit["complete_panel"] and audit["records"] == 96
    result = selection(protocol, rows)
    assert result["winners"] == {
        "tabu": "tabu/fixed_h5",
        "adaptive": "adaptive/explore_stagnation10",
    }
    assert result["selected_configuration"]["tabu_adaptive_penalty"] is False
    assert result["selected_configuration"]["adaptive_exploration"] == 0.4
    assert result["selected_configuration"]["stagnation_repairs"] == 10
    with pytest.raises(ValueError, match="complete tuning"):
        selection(protocol, rows[:-1])
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="failures"):
        selection(protocol, rows)


def test_tuning_selection_uses_60_second_results_before_secondary_scores(tmp_path):
    _, protocol, rows = tuning_panel(tmp_path)
    for row in rows:
        # This synthetic score panel tests only the selection rule, not graph evidence.
        if row["trial"] == "tabu/reactive_h5":
            row["checkpoints"] = {str(t): {"edges": 1 if t == 60 else 0} for t in [1, 10, 60]}
        elif row["method"] == "tabu":
            row["checkpoints"] = {str(t): {"edges": 0 if t == 60 else 100} for t in [1, 10, 60]}
    assert selection(protocol, rows)["winners"]["tabu"] == "tabu/reactive_h5"


def test_tabu_really_bans_the_next_declared_number_of_moves(monkeypatch):
    # n4 has no feasible graph above four edges, so aspiration cannot override
    # tenure here. Spy on actual flips while the real tabu transition code runs.
    graph = Graph.from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2)])
    ticker = itertools.count()
    monkeypatch.setattr(search.time, "perf_counter", lambda: next(ticker) * 0.001)

    def initialize(n, seed, clock):
        clock.offer(graph, "test_start")
        return [SimpleNamespace(graph=graph, kind="test", field_order=None)]

    monkeypatch.setattr(search, "initialize", initialize)
    actions = []
    original_add, original_remove = Graph.add, Graph.remove

    def add(self, u, v):
        actions.append((u, v))
        return original_add(self, u, v)

    def remove(self, u, v):
        actions.append((u, v))
        return original_remove(self, u, v)

    monkeypatch.setattr(Graph, "add", add)
    monkeypatch.setattr(Graph, "remove", remove)
    result = search.tabu_search(
        4,
        17,
        SearchConfig(
            seconds=1,
            tabu_tenure_min=3,
            tabu_tenure_max=3,
            tabu_restart_iterations=1000000,
            tabu_adaptive_penalty=False,
        ),
    )
    assert len(actions) > 10 and result["iterations"] == len(actions)
    assert all(edge not in actions[max(0, i - 3) : i] for i, edge in enumerate(actions))


def test_invalid_adaptive_search_configuration_is_rejected():
    with pytest.raises(ValueError, match="adaptive"):
        SearchConfig(adaptive_exploration=0)
    assert asdict(SearchConfig())["tabu_adaptive_penalty"] is True


def test_retained_selection_cannot_relabel_winner_or_provenance(tmp_path):
    directory, _, _ = tuning_panel(tmp_path)
    report_tuning(SimpleNamespace(batch=directory))
    path = directory / "SELECTION.json"
    original = json.loads(path.read_text())
    assert load_tuning(path).tabu_tenure_min == 5
    changed = {**original, "winners": {**original["winners"], "tabu": "tabu/reactive_h5"}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="predeclared"):
        load_tuning(path)
    path.write_text(json.dumps({**original, "protocol_sha256": "changed"}))
    with pytest.raises(ValueError, match="provenance"):
        load_tuning(path)


@pytest.mark.parametrize("position", [-1, 8, "0"])
def test_invalid_tuning_order_position_is_rejected(tmp_path, position):
    directory, _, rows = tuning_panel(tmp_path)
    rows[0]["trial_position"] = position
    (directory / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="invalid tuning position"):
        independent_audit(directory)

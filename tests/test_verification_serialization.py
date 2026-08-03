import json

import pytest

from extremal_graph import GraphState, load_graph, save_graph, verify_graph
from extremal_graph.serialization import graph_from_dict, graph_to_dict


def test_verifier_distinguishes_constraint_and_maximality_failures() -> None:
    triangle = GraphState(n=4, r=2, edges=((0, 1), (1, 2), (0, 2)))
    report = verify_graph(triangle)
    assert not report.constraint_free
    assert not report.valid
    assert report.forbidden_cliques == ((0, 1, 2),)

    nonmaximal = GraphState(n=4, r=2, edges=((0, 1),))
    report = verify_graph(nonmaximal)
    assert report.constraint_free
    assert not report.maximal
    assert report.addable_edges


def test_graph_json_round_trip(tmp_path) -> None:
    graph = GraphState(n=5, r=2, edges=((3, 1), (4, 0)))
    path = tmp_path / "nested" / "graph.json"
    save_graph(graph, path)
    assert load_graph(path) == graph
    assert json.loads(path.read_text()) == graph_to_dict(graph)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"schema_version": 2, "n": 3, "r": 2, "edges": []},
        {"schema_version": 1, "n": 3, "r": 2, "edges": [[0]]},
        {"schema_version": 1, "n": 3, "r": 2, "edges": [], "extra": True},
    ],
)
def test_invalid_serialized_values(value) -> None:
    with pytest.raises(ValueError):
        graph_from_dict(value)

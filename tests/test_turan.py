from itertools import combinations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from extremal_graph import construct_turan, turan_edge_count, turan_partition_sizes
from extremal_graph.verification import verify_graph


@given(n=st.integers(min_value=0, max_value=80), r=st.integers(min_value=1, max_value=12))
def test_partition_and_edge_formula_match_construction(n: int, r: int) -> None:
    sizes = turan_partition_sizes(n, r)
    graph = construct_turan(n, r)
    assert len(sizes) == r
    assert sum(sizes) == n
    assert max(sizes, default=0) - min(sizes, default=0) <= 1
    assert graph.edge_count == turan_edge_count(n, r)


@pytest.mark.parametrize(
    ("n", "r", "expected"),
    [(2, 2, 1), (3, 2, 2), (4, 2, 4), (5, 2, 6), (6, 2, 9), (7, 3, 16)],
)
def test_known_turan_edge_counts(n: int, r: int, expected: int) -> None:
    assert turan_edge_count(n, r) == expected


def test_constructed_graph_is_clique_free_and_maximal() -> None:
    for n, r in ((6, 2), (8, 3), (9, 4)):
        graph = construct_turan(n, r)
        report = verify_graph(graph, r=r)
        assert report.valid
        edge_set = set(graph.edges)
        assert not any(
            all((min(u, v), max(u, v)) in edge_set for u, v in combinations(nodes, 2))
            for nodes in combinations(range(n), r + 1)
        )


@pytest.mark.parametrize(("n", "r"), [(-1, 2), (3, 0), (3, -1)])
def test_invalid_turan_arguments(n: int, r: int) -> None:
    with pytest.raises(ValueError):
        turan_edge_count(n, r)

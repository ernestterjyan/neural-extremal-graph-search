import itertools
import random

import pytest

from experiments.c4_solver_pilot import (
    add,
    edges,
    legal_edges,
    polarity_graph,
    solve,
    verify_graph,
)


def brute_legal(adjacency):
    n = len(adjacency)
    result = []
    for u, v in itertools.combinations(range(n), 2):
        if adjacency[u] & (1 << v):
            continue
        trial = adjacency.copy()
        add(trial, (u, v))
        graph_edges = set(edges(trial))
        has_c4 = any(
            all(edge in graph_edges for edge in cycle)
            for four in itertools.combinations(range(n), 4)
            for cycle in [
                ((four[0], four[1]), (four[1], four[2]), (four[2], four[3]), (four[0], four[3])),
                ((four[0], four[1]), (four[1], four[3]), (four[2], four[3]), (four[0], four[2])),
                ((four[0], four[2]), (four[1], four[2]), (four[1], four[3]), (four[0], four[3])),
            ]
        )
        if not has_c4:
            result.append((u, v))
    return result


def test_bitset_legality_agrees_with_four_vertex_enumeration():
    rng = random.Random(47)
    for n in range(4, 8):
        for _ in range(12):
            adjacency = [0] * n
            for _ in range(8):
                assert legal_edges(adjacency) == brute_legal(adjacency)
                available = legal_edges(adjacency)
                if not available:
                    break
                add(adjacency, rng.choice(available))


@pytest.mark.parametrize("q,expected", [(2, 9), (3, 24), (5, 90), (7, 224)])
def test_polarity_construction_is_c4_free(q, expected):
    adjacency = polarity_graph(q)
    assert len(adjacency) == q * q + q + 1
    assert len(edges(adjacency)) == expected
    # The independent graph verifier also checks maximality at these full orders.
    verify_graph(len(adjacency), edges(adjacency))


@pytest.mark.parametrize(
    "method", ["uniform", "min_degree", "lookahead16", "swap200", "polarity", "polarity_swap200"]
)
def test_pilot_methods_return_reproducible_valid_graphs(method):
    first = solve(10, method, 123)
    second = solve(10, method, 123)
    assert first["edges"] == second["edges"]
    verify_graph(10, first["edges"])

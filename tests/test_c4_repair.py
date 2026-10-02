"""Independent combinatorial checks for the new repair campaign."""

import itertools
import random

import pytest

from extremal_graph.repair.constructions import FiniteField, polarity_graph, starting_pool
from extremal_graph.repair.engine import repair
from extremal_graph.repair.graph import Graph, cycle_cuts, verify_witness
from extremal_graph.repair.regions import candidate_regions
from extremal_graph.repair.search import SearchClock, SearchConfig, neighborhood_search, tabu_search


def explicit_count(n, edges):
    edge_set = {frozenset(e) for e in edges}
    return sum(
        all(frozenset(e) in edge_set for e in [(w, x), (x, y), (y, z), (z, w)])
        for a, b, c, d in itertools.combinations(range(n), 4)
        for w, x, y, z in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]
    )


def enumerate_graphs(n):
    pairs = list(itertools.combinations(range(n), 2))
    for bits in range(1 << len(pairs)):
        edges = [e for j, e in enumerate(pairs) if bits & (1 << j)]
        yield edges


def test_updates_and_rollback_exhaustive():
    for edges in enumerate_graphs(5):
        graph = Graph.from_edges(5, edges)
        before = graph.adjacency.copy()
        count = explicit_count(5, edges)
        assert graph.c4_count() == count == len(cycle_cuts(graph))
        for u, v in itertools.combinations(range(5), 2):
            if graph.has(u, v):
                delta = graph.deletion_cycles(u, v)
                graph.remove(u, v)
                assert graph.c4_count() == count - delta
                graph.add(u, v)
            else:
                delta = graph.addition_cycles(u, v)
                graph.add(u, v)
                assert explicit_count(5, graph.edges()) == count + delta
                graph.remove(u, v)
            assert graph.adjacency == before
        if count == 0:
            verify_witness(5, edges, explicit_cycles=True)
            assert set(graph.legal_edges()) == {
                e
                for e in itertools.combinations(range(5), 2)
                if not graph.has(*e) and explicit_count(5, [*edges, e]) == 0
            }
        else:
            with pytest.raises(ValueError, match="C4"):
                verify_witness(5, edges)


@pytest.mark.parametrize("q", [2, 3, 4, 5, 7, 8, 9, 11, 16])
def test_finite_field_axioms_and_polarity(q):
    field = FiniteField(q)
    add, mul = field.add_table, field.mul_table
    for a in range(q):
        assert add[a][0] == a and mul[a][1] == a and mul[a][0] == 0
        assert any(add[a][b] == 0 for b in range(q))
        if a:
            assert any(mul[a][b] == 1 for b in range(q))
        for b in range(q):
            assert add[a][b] == add[b][a] and mul[a][b] == mul[b][a]
            for c in range(q):
                assert add[add[a][b]][c] == add[a][add[b][c]]
                assert mul[mul[a][b]][c] == mul[a][mul[b][c]]
                assert mul[a][add[b][c]] == add[mul[a][b]][mul[a][c]]
    graph = polarity_graph(q)
    assert graph.n == q * q + q + 1
    assert graph.m == q * (q + 1) ** 2 // 2
    verify_witness(graph.n, graph.edges())


def test_no_composite_modular_arithmetic():
    with pytest.raises(ValueError, match="prime power"):
        FiniteField(6)


def test_exact_repair_exhaustive_order_four():
    # All feasible graphs and all 15 nonempty regions, with a separate enumeration
    # of every feasible completion preserving the outside edges.
    feasible = [edges for edges in enumerate_graphs(4) if explicit_count(4, edges) == 0]
    for edges in feasible:
        graph = Graph.from_edges(4, edges)
        for bits in range(1, 16):
            selected = tuple(u for u in range(4) if bits & (1 << u))
            outside = set(range(4)) - set(selected)
            fixed = {e for e in edges if set(e) <= outside}
            optimum = max(
                len(es) for es in feasible if {e for e in es if set(e) <= outside} == fixed
            )
            result = repair(graph, selected, seconds=2, seed=123)
            assert result.graph.m == optimum
            assert result.local_optimal
            assert result.global_optimal == (len(selected) == 4)
            assert result.graph.m >= graph.m
            assert graph.edges() == edges
            verify_witness(4, result.graph.edges(), explicit_cycles=True)


def test_exact_repair_order_five_and_timeout():
    graph = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
    result = repair(graph, tuple(range(5)), seconds=5, seed=33)
    feasible = [edges for edges in enumerate_graphs(5) if explicit_count(5, edges) == 0]
    assert result.graph.m == max(map(len, feasible))
    assert result.global_optimal
    timeout = repair(graph, (1, 3), seconds=0, seed=33)
    assert timeout.graph.adjacency == graph.adjacency
    assert not timeout.local_optimal and not timeout.global_optimal


def test_work_limited_replay_and_permutation():
    graph = polarity_graph(3)
    a = repair(graph, (0, 1, 2), seconds=10, seed=7, deterministic_limit=0.01)
    b = repair(graph, (0, 1, 2), seconds=10, seed=7, deterministic_limit=0.01)
    assert a.graph.digest() == b.graph.digest()
    assert (a.rounds, a.cuts, a.status) == (b.rounds, b.cuts, b.status)
    small = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    order = [4, 2, 0, 3, 1]
    permuted = small.induced(order)
    selected = (1, 3)
    mapped = tuple(order.index(u) for u in selected)
    original = repair(small, selected, seconds=5, seed=8)
    transformed = repair(permuted, mapped, seconds=5, seed=8)
    assert original.local_optimal and transformed.local_optimal
    assert original.graph.m == transformed.graph.m


def test_starting_pool_and_regions_replay():
    first = starting_pool(35, 999)
    second = starting_pool(35, 999)
    assert [s.graph.digest() for s in first] == [s.graph.digest() for s in second]
    assert {s.kind for s in first} == {"trim_polarity", "extend_polarity", "min_degree", "uniform"}
    for start in first:
        verify_witness(35, start.graph.edges())
    graph = first[0].graph
    a = candidate_regions(graph, random.Random(22))
    b = candidate_regions(graph, random.Random(22))
    assert a == b
    assert {r.family for r in a} == {"random", "neighborhood", "blocking_path"}
    assert len({r.vertices for r in a}) == len(a)
    assert all(len(r.vertices) in {3, 5, 7} for r in a)


def test_invalid_witnesses_rejected():
    for edges in [[(0, 0)], [(0, 1), (0, 1)], [(1, 0)], [(0, 4)]]:
        with pytest.raises(ValueError):
            verify_witness(4, edges)


def test_checkpoint_accounting_and_timed_search():
    clock = SearchClock(4, 60)
    one = Graph.from_edges(4, [(0, 1)])
    two = Graph.from_edges(4, [(0, 1), (1, 2)])
    clock.offer(one, "early", completed_at=clock.started + 0.5)
    clock.offer(two, "late", completed_at=clock.started + 1.5)
    clock.offer(
        Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)]), "outside", completed_at=clock.started + 60.01
    )
    result = clock.report()
    assert result["checkpoints"]["1"]["edges"] == 1
    assert result["checkpoints"]["10"]["edges"] == 2
    assert result["edge_count"] == 2
    config = SearchConfig(seconds=0.2, region_sizes=(2, 3), repair_seconds=0.02)
    results = [tabu_search(12, 717, config), neighborhood_search(12, 717, config)]
    assert results[0]["starting_pool"] == results[1]["starting_pool"]
    for outcome in results:
        verify_witness(12, outcome["edges"], explicit_cycles=True)
        assert all(event["seconds"] <= 0.2 for event in outcome["history"])
        assert outcome["initialization_seconds"] > 0


def test_selectors_are_permutation_equivariant():
    import torch

    from extremal_graph.repair.selectors import features, make_model

    torch.set_num_threads(1)
    graph = polarity_graph(3)
    regions = candidate_regions(graph, random.Random(23), sizes=(3, 5), per_family=1)
    order = list(range(graph.n))
    random.Random(55).shuffle(order)
    inverse = {old: new for new, old in enumerate(order)}
    permuted = graph.induced(order)
    mapped = [type(r)(tuple(inverse[u] for u in r.vertices), r.family) for r in regions]
    for family in ["gnn", "mlp"]:
        torch.manual_seed(411)
        model = make_model(family)
        a, b, c = features(graph, regions)
        x, y, z = features(permuted, mapped)
        torch.testing.assert_close(c, z, atol=1e-6, rtol=1e-6)
        original = model(a, b, c, regions)
        transformed = model(x, y, z, mapped)
        torch.testing.assert_close(original, transformed, atol=1e-6, rtol=1e-6)

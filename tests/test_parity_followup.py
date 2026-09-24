"""Independent checks of the preregistered balancing control."""

import random
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import networkx as nx

from extremal_graph.env import GraphConstructionEnv
from extremal_graph.features import legal_edges_from_state
from extremal_graph.graph import GraphState
from extremal_graph.study import independent_verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from parity_followup import balance_scores, generate_balance  # noqa: E402


def test_balance_score_matches_networkx_after_every_small_state_action():
    for n in range(4, 9):
        env = GraphConstructionEnv(n)
        rng = random.Random(n)
        for _ in range(n + 2):
            graph = nx.empty_graph(n)
            graph.add_edges_from(env.state.edges)
            if not nx.is_bipartite(graph):
                break
            legal = env.legal_edges()
            if not legal:
                break
            scores = balance_scores(env.state, legal)
            for edge, score in zip(legal, scores, strict=True):
                augmented = graph.copy()
                augmented.add_edge(*edge)
                if not nx.is_bipartite(augmented):
                    assert score is None
                    continue
                colored = nx.bipartite.color(augmented)
                component = nx.node_connected_component(augmented, edge[0])
                zeros = sum(colored[v] == 0 for v in component)
                ones = len(component) - zeros
                assert score == abs(zeros - ones)
            env.step(rng.choice(legal))


def test_balancing_trajectories_are_batch_order_independent_and_valid():
    seeds = [307, 311, 313]
    together = generate_balance(8, seeds)
    reversed_results = generate_balance(8, list(reversed(seeds)))
    for expected, actual in zip(together, reversed(reversed_results), strict=True):
        assert expected.actions == actual.actions
        assert all(independent_verify(actual.final_state).values())
        graph = nx.empty_graph(8)
        graph.add_edges_from(actual.final_state.edges)
        assert nx.is_bipartite(graph)
    assert together[0].actions == generate_balance(8, seeds[:1])[0].actions


def test_all_balancing_tie_choices_reach_optimum_through_six_vertices():
    for n in range(2, 7):
        initial = GraphState(n, 2)
        queue = deque([initial])
        seen = {initial.edges}
        while queue:
            state = queue.popleft()
            legal = legal_edges_from_state(state)
            if not legal:
                assert state.edge_count == n * n // 4
                continue
            scores = balance_scores(state, legal)
            best = min(value for value in scores if value is not None)
            for edge, value in zip(legal, scores, strict=True):
                if value != best:
                    continue
                child = GraphState(n, 2, state.edges + (edge,))
                assert any(
                    all((u in side) != (v in side) for u, v in child.edges)
                    for side_tuple in combinations(range(n), n // 2)
                    for side in [set(side_tuple)]
                )
                if child.edges not in seen:
                    seen.add(child.edges)
                    queue.append(child)

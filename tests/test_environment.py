import math
from itertools import combinations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from extremal_graph import (
    GraphConstructionEnv,
    GraphState,
    InvalidActionError,
    LeastDegreePolicy,
    UniformRandomPolicy,
    run_baseline_episode,
    turan_edge_count,
    verify_graph,
)


def _contains_triangle(n: int, edges: set[tuple[int, int]]) -> bool:
    return any(
        all((min(u, v), max(u, v)) in edges for u, v in combinations(vertices, 2))
        for vertices in combinations(range(n), 3)
    )


def _brute_legal(state: GraphState) -> list[tuple[int, int]]:
    existing = set(state.edges)
    legal = []
    for u in range(state.n):
        for v in range(u + 1, state.n):
            if (u, v) in existing:
                continue
            if not _contains_triangle(state.n, existing | {(u, v)}):
                legal.append((u, v))
    return legal


def test_fast_legality_matches_brute_force_for_all_triangle_free_graphs_through_n6() -> None:
    for n in range(2, 7):
        possible = list(combinations(range(n), 2))
        for mask in range(1 << len(possible)):
            edges = {possible[index] for index in range(len(possible)) if mask & (1 << index)}
            if _contains_triangle(n, edges):
                continue
            state = GraphState(n=n, r=2, edges=tuple(edges))
            env = GraphConstructionEnv.from_state(state)
            assert env.legal_edges() == _brute_legal(state)


@given(
    n=st.integers(min_value=2, max_value=24),
    seed=st.integers(min_value=0, max_value=1_000_000),
)
@settings(max_examples=40, deadline=None)
def test_random_episodes_are_valid_terminal_graphs(n: int, seed: int) -> None:
    trajectory = run_baseline_episode(n, UniformRandomPolicy(), seed=seed)
    report = verify_graph(trajectory.final_state)
    assert report.valid
    assert math.isclose(
        trajectory.return_value,
        trajectory.final_state.edge_count / turan_edge_count(n, 2),
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_least_degree_and_random_are_reproducible() -> None:
    for policy in (UniformRandomPolicy(), LeastDegreePolicy()):
        first = run_baseline_episode(12, policy, seed=77)
        second = run_baseline_episode(12, policy, seed=77)
        assert first.actions == second.actions
        assert first.final_state == second.final_state


def test_environment_rejects_invalid_actions_and_parameters() -> None:
    with pytest.raises(ValueError):
        GraphConstructionEnv(n=1)
    with pytest.raises(ValueError):
        GraphConstructionEnv(n=5, r=3)

    env = GraphConstructionEnv(n=4)
    env.step((0, 1))
    with pytest.raises(InvalidActionError):
        env.step((1, 0))
    with pytest.raises(InvalidActionError):
        env.step((2, 2))
    env.step((1, 2))
    with pytest.raises(InvalidActionError):
        env.step((0, 2))


def test_from_state_rejects_existing_triangle() -> None:
    graph = GraphState(n=3, r=2, edges=((0, 1), (1, 2), (0, 2)))
    with pytest.raises(ValueError):
        GraphConstructionEnv.from_state(graph)

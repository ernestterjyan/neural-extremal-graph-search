"""Exact balanced-completion diagnostics, never supplied to primary policies."""

from dataclasses import dataclass

from .graph import Edge, GraphState, Trajectory
from .turan import turan_edge_count


@dataclass
class Components:
    colors: list[int]
    component: list[int]
    sizes: list[tuple[int, int]]
    bipartite: bool


def components(state: GraphState) -> Components:
    adjacency = state.adjacency_sets()
    colors = [-1] * state.n
    membership = [-1] * state.n
    sizes = []
    for root in range(state.n):
        if colors[root] >= 0:
            continue
        index = len(sizes)
        colors[root] = 0
        membership[root] = index
        counts = [1, 0]
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if colors[v] < 0:
                    colors[v] = 1 - colors[u]
                    membership[v] = index
                    counts[colors[v]] += 1
                    stack.append(v)
                elif colors[v] == colors[u]:
                    return Components(colors, membership, sizes, False)
        sizes.append((counts[0], counts[1]))
    return Components(colors, membership, sizes, True)


def _can_balance(sizes: list[tuple[int, int]], n: int) -> bool:
    reachable = 1
    mask = (1 << (n // 2 + 1)) - 1
    for a, b in sizes:
        reachable = ((reachable << a) | (reachable << b)) & mask
    return bool(reachable & (1 << (n // 2)))


def balanced_completion(state: GraphState) -> tuple[bool, str]:
    if state.r != 2:
        raise ValueError("balanced bipartite completion is defined for r=2")
    value = components(state)
    if not value.bipartite:
        return False, "odd_cycle"
    if not _can_balance(value.sizes, state.n):
        return False, "partition_imbalance"
    return True, "feasible"


def preserving_actions(state: GraphState, legal: list[Edge]) -> list[bool]:
    """Whether each action preserves some balanced complete bipartite completion."""
    value = components(state)
    if not value.bipartite or not _can_balance(value.sizes, state.n):
        return [False] * len(legal)
    cache = {}
    result = []
    for u, v in legal:
        i, j = value.component[u], value.component[v]
        flip = value.colors[u] ^ value.colors[v] ^ 1
        if i == j:
            result.append(flip == 0)
            continue
        key = (min(i, j), max(i, j), flip)
        if key not in cache:
            a, b = value.sizes[i]
            c, d = value.sizes[j]
            merged = (a + c, b + d) if flip == 0 else (a + d, b + c)
            remaining = [pair for k, pair in enumerate(value.sizes) if k not in (i, j)]
            cache[key] = _can_balance([*remaining, merged], state.n)
        result.append(cache[key])
    return result


def trajectory_diagnostics(trajectory: Trajectory) -> dict:
    state = trajectory.final_state
    parts = components(state)
    exact = state.edge_count == turan_edge_count(state.n, 2)
    outcome = "exact" if exact else "unbalanced_bipartite" if parts.bipartite else "nonbipartite"
    first_loss = None
    cause = None
    first_odd = None
    edges = []
    for step, edge in enumerate(trajectory.actions, 1):
        edges.append(edge)
        feasible, why = balanced_completion(GraphState(state.n, 2, tuple(edges)))
        if not feasible and first_loss is None:
            first_loss, cause = step, why
        if why == "odd_cycle":
            first_odd = step
            break
    return {
        "outcome": outcome,
        "bipartite": parts.bipartite,
        "first_loss_step": first_loss,
        "first_loss_cause": cause,
        "first_odd_cycle_step": first_odd,
    }

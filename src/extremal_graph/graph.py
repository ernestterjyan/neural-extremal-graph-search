"""Core immutable graph values shared by the environment and model."""

from __future__ import annotations

from dataclasses import dataclass

type Edge = tuple[int, int]


def canonical_edge(u: int, v: int, n: int | None = None) -> Edge:
    """Return an undirected edge in canonical order after validating it."""
    if (
        isinstance(u, bool)
        or isinstance(v, bool)
        or not isinstance(u, int)
        or not isinstance(v, int)
    ):
        raise TypeError("edge endpoints must be integers")
    if u == v:
        raise ValueError("self-loops are not allowed")
    if n is not None and not (0 <= u < n and 0 <= v < n):
        raise ValueError(f"edge endpoints must lie in [0, {n})")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True, slots=True)
class GraphState:
    """An immutable snapshot of a simple undirected graph."""

    n: int
    r: int
    edges: tuple[Edge, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("n must be a non-negative integer")
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r < 1:
            raise ValueError("r must be a positive integer")

        canonical = tuple(sorted(canonical_edge(u, v, self.n) for u, v in self.edges))
        if len(set(canonical)) != len(canonical):
            raise ValueError("duplicate edges are not allowed")
        object.__setattr__(self, "edges", canonical)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def degrees(self) -> tuple[int, ...]:
        values = [0] * self.n
        for u, v in self.edges:
            values[u] += 1
            values[v] += 1
        return tuple(values)

    def adjacency_sets(self) -> tuple[frozenset[int], ...]:
        neighbours = [set() for _ in range(self.n)]
        for u, v in self.edges:
            neighbours[u].add(v)
            neighbours[v].add(u)
        return tuple(frozenset(values) for values in neighbours)


@dataclass(frozen=True, slots=True)
class Transition:
    """Result of one legal graph-building action."""

    state: GraphState
    action: Edge
    reward: float
    terminated: bool


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A completed episode suitable for training or analysis."""

    n: int
    r: int
    states: tuple[GraphState, ...]
    actions: tuple[Edge, ...]
    rewards: tuple[float, ...]
    final_state: GraphState
    optimality_ratio: float

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) or len(self.actions) != len(self.rewards):
            raise ValueError("states, actions, and rewards must have equal lengths")

    @property
    def return_value(self) -> float:
        return sum(self.rewards)

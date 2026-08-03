"""Triangle-free monotone graph-building environment."""

from __future__ import annotations

import random

from .graph import Edge, GraphState, Transition, canonical_edge
from .turan import turan_edge_count


class InvalidActionError(ValueError):
    """Raised when an edge cannot legally be added to the current graph."""


class GraphConstructionEnv:
    """Build a maximal triangle-free graph by adding one legal edge at a time."""

    def __init__(self, n: int | None = None, r: int = 2, seed: int | None = None) -> None:
        self._n = 0
        self._r = 2
        self._edges: set[Edge] = set()
        self._neighbours: list[int] = []
        self._rng = random.Random()
        self._seed: int | None = None
        if n is not None:
            self.reset(n=n, r=r, seed=seed)

    def reset(self, n: int, r: int = 2, seed: int | None = None) -> GraphState:
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("the MVP environment requires integer n >= 2")
        if r != 2:
            raise ValueError("the MVP environment supports only r = 2")
        self._n = n
        self._r = r
        self._edges = set()
        self._neighbours = [0] * n
        self._seed = seed
        self._rng.seed(seed)
        return self.state

    @classmethod
    def from_state(
        cls,
        state: GraphState,
        *,
        seed: int | None = None,
    ) -> GraphConstructionEnv:
        """Restore a valid triangle-free state, primarily for analysis and tests."""
        environment = cls(n=state.n, r=state.r, seed=seed)
        for u, v in state.edges:
            environment._edges.add((u, v))
            environment._neighbours[u] |= 1 << v
            environment._neighbours[v] |= 1 << u
        for u, v in state.edges:
            if environment._neighbours[u] & environment._neighbours[v]:
                raise ValueError("cannot restore a state that already contains a triangle")
        return environment

    @property
    def n(self) -> int:
        return self._n

    @property
    def r(self) -> int:
        return self._r

    @property
    def state(self) -> GraphState:
        self._require_reset()
        return GraphState(n=self._n, r=self._r, edges=tuple(self._edges))

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def terminated(self) -> bool:
        return not self.legal_edges()

    @property
    def rng(self) -> random.Random:
        return self._rng

    def _require_reset(self) -> None:
        if self._n < 2:
            raise RuntimeError("reset the environment before use")

    def is_legal(self, edge: Edge) -> bool:
        self._require_reset()
        try:
            u, v = canonical_edge(*edge, self._n)
        except (TypeError, ValueError):
            return False
        return (u, v) not in self._edges and not (self._neighbours[u] & self._neighbours[v])

    def legal_edges(self) -> list[Edge]:
        self._require_reset()
        legal: list[Edge] = []
        for u in range(self._n):
            for v in range(u + 1, self._n):
                if (u, v) not in self._edges and not (self._neighbours[u] & self._neighbours[v]):
                    legal.append((u, v))
        return legal

    def legal_incident_counts(self) -> tuple[int, ...]:
        counts = [0] * self._n
        for u, v in self.legal_edges():
            counts[u] += 1
            counts[v] += 1
        return tuple(counts)

    def neighbour_masks(self) -> tuple[int, ...]:
        self._require_reset()
        return tuple(self._neighbours)

    def step(self, edge: Edge) -> Transition:
        self._require_reset()
        try:
            action = canonical_edge(*edge, self._n)
        except (TypeError, ValueError) as error:
            raise InvalidActionError(str(error)) from error
        if not self.is_legal(action):
            raise InvalidActionError(f"edge {action} is not legal in the current state")

        u, v = action
        self._edges.add(action)
        self._neighbours[u] |= 1 << v
        self._neighbours[v] |= 1 << u
        reward = 1.0 / turan_edge_count(self._n, self._r)
        return Transition(
            state=self.state,
            action=action,
            reward=reward,
            terminated=self.terminated,
        )

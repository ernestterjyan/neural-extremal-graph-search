"""Independent, deliberately simple graph verification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .graph import Edge, GraphState


@dataclass(frozen=True, slots=True)
class VerificationReport:
    n: int
    r: int
    constraint_free: bool
    maximal: bool
    edge_count: int
    forbidden_cliques: tuple[tuple[int, ...], ...]
    addable_edges: tuple[Edge, ...]

    @property
    def valid(self) -> bool:
        return self.constraint_free and self.maximal


def _is_clique(vertices: tuple[int, ...], edges: set[Edge]) -> bool:
    return all((min(u, v), max(u, v)) in edges for u, v in combinations(vertices, 2))


def _forbidden_cliques(graph: GraphState, r: int) -> tuple[tuple[int, ...], ...]:
    edges = set(graph.edges)
    return tuple(
        vertices for vertices in combinations(range(graph.n), r + 1) if _is_clique(vertices, edges)
    )


def _adding_edge_is_constraint_free(graph: GraphState, edge: Edge, r: int) -> bool:
    augmented = set(graph.edges)
    augmented.add(edge)
    return not any(
        _is_clique(vertices, augmented) for vertices in combinations(range(graph.n), r + 1)
    )


def verify_graph(
    graph: GraphState,
    r: int | None = None,
    *,
    require_maximal: bool = True,
) -> VerificationReport:
    """Verify clique-freeness and, optionally, terminal maximality.

    This implementation uses explicit combinations and edge membership. It does
    not call the bitset legality path used by :class:`GraphConstructionEnv`.
    """
    checked_r = graph.r if r is None else r
    if checked_r < 1:
        raise ValueError("r must be positive")

    forbidden = _forbidden_cliques(graph, checked_r)
    existing = set(graph.edges)
    addable: list[Edge] = []
    if not forbidden and require_maximal:
        if checked_r == 2:
            neighbours = [set() for _ in range(graph.n)]
            for u, v in graph.edges:
                neighbours[u].add(v)
                neighbours[v].add(u)
            for u in range(graph.n):
                for v in range(u + 1, graph.n):
                    edge = (u, v)
                    if edge not in existing and neighbours[u].isdisjoint(neighbours[v]):
                        addable.append(edge)
        else:
            for u in range(graph.n):
                for v in range(u + 1, graph.n):
                    edge = (u, v)
                    if edge not in existing and _adding_edge_is_constraint_free(
                        graph, edge, checked_r
                    ):
                        addable.append(edge)

    maximal = not addable if require_maximal else True
    return VerificationReport(
        n=graph.n,
        r=checked_r,
        constraint_free=not forbidden,
        maximal=maximal,
        edge_count=graph.edge_count,
        forbidden_cliques=forbidden,
        addable_edges=tuple(addable),
    )

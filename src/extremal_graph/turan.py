"""Exact Turán graph formulas and construction."""

from __future__ import annotations

from .graph import Edge, GraphState


def _validate(n: int, r: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise ValueError("r must be a positive integer")


def turan_partition_sizes(n: int, r: int) -> tuple[int, ...]:
    """Return the balanced part sizes of the Turán graph ``T_r(n)``."""
    _validate(n, r)
    quotient, remainder = divmod(n, r)
    return (quotient + 1,) * remainder + (quotient,) * (r - remainder)


def turan_edge_count(n: int, r: int) -> int:
    """Return ``ex(n, K_{r+1})``, the number of edges in ``T_r(n)``."""
    sizes = turan_partition_sizes(n, r)
    return (n * n - sum(size * size for size in sizes)) // 2


def construct_turan(n: int, r: int) -> GraphState:
    """Construct a canonical balanced complete ``r``-partite graph."""
    sizes = turan_partition_sizes(n, r)
    part_by_vertex: list[int] = []
    for part, size in enumerate(sizes):
        part_by_vertex.extend([part] * size)

    edges: list[Edge] = []
    for u in range(n):
        for v in range(u + 1, n):
            if part_by_vertex[u] != part_by_vertex[v]:
                edges.append((u, v))
    return GraphState(n=n, r=r, edges=tuple(edges))

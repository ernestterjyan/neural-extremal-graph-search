"""Bitset graph operations and an independent, set-based witness verifier."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass

Edge = tuple[int, int]


def vertices(bits: int) -> Iterable[int]:
    while bits:
        bit = bits & -bits
        yield bit.bit_length() - 1
        bits ^= bit


@dataclass
class Graph:
    adjacency: list[int]

    @classmethod
    def empty(cls, n: int) -> Graph:
        if n < 0:
            raise ValueError("negative order")
        return cls([0] * n)

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Edge]) -> Graph:
        result = cls.empty(n)
        for u, v in edges:
            result.add(u, v)
        return result

    @property
    def n(self) -> int:
        return len(self.adjacency)

    @property
    def m(self) -> int:
        return sum(row.bit_count() for row in self.adjacency) // 2

    def copy(self) -> Graph:
        return Graph(self.adjacency.copy())

    def has(self, u: int, v: int) -> bool:
        return bool(self.adjacency[u] & (1 << v))

    def add(self, u: int, v: int) -> None:
        if not 0 <= u < self.n or not 0 <= v < self.n or u == v or self.has(u, v):
            raise ValueError("invalid or duplicate edge")
        self.adjacency[u] |= 1 << v
        self.adjacency[v] |= 1 << u

    def remove(self, u: int, v: int) -> None:
        if not self.has(u, v):
            raise ValueError("missing edge")
        self.adjacency[u] &= ~(1 << v)
        self.adjacency[v] &= ~(1 << u)

    def edges(self) -> list[Edge]:
        return [(u, v) for u, row in enumerate(self.adjacency) for v in vertices(row) if u < v]

    def digest(self) -> str:
        return hashlib.sha256(json.dumps([self.n, self.edges()]).encode()).hexdigest()

    def c4_count(self) -> int:
        # Each C4 has two opposite vertex pairs, including in K4.
        return (
            sum(
                count * (count - 1) // 2
                for u, v in itertools.combinations(range(self.n), 2)
                for count in [(self.adjacency[u] & self.adjacency[v]).bit_count()]
            )
            // 2
        )

    def addition_cycles(self, u: int, v: int) -> int:
        """Number of length-three paths: the exact C4 increase upon insertion."""
        if u == v or self.has(u, v):
            raise ValueError("addition requires a missing edge")
        return sum(
            (self.adjacency[a] & self.adjacency[v]).bit_count() for a in vertices(self.adjacency[u])
        )

    def deletion_cycles(self, u: int, v: int) -> int:
        if not self.has(u, v):
            raise ValueError("deletion requires an edge")
        return sum(
            (self.adjacency[a] & self.adjacency[v] & ~(1 << u)).bit_count()
            for a in vertices(self.adjacency[u] & ~(1 << v))
        )

    def legal_edges(self) -> list[Edge]:
        two_hop = [0] * self.n
        for u, row in enumerate(self.adjacency):
            for a in vertices(row):
                two_hop[u] |= self.adjacency[a]
        return [
            (u, v)
            for u, v in itertools.combinations(range(self.n), 2)
            if not self.has(u, v) and not two_hop[u] & self.adjacency[v]
        ]

    def induced(self, order: Iterable[int]) -> Graph:
        order = list(order)
        if len(set(order)) != len(order) or any(not 0 <= u < self.n for u in order):
            raise ValueError("invalid vertex order")
        return Graph.from_edges(
            len(order),
            (
                (i, j)
                for i, u in enumerate(order)
                for j, v in enumerate(order)
                if i < j and self.has(u, v)
            ),
        )


def greedy_fill(
    graph: Graph, rng: random.Random, *, deadline: float = float("inf"), method: str = "min_degree"
) -> Graph:
    while time.perf_counter() < deadline:
        candidates = graph.legal_edges()
        if not candidates:
            break
        if method == "uniform":
            edge = rng.choice(candidates)
        elif method == "min_degree":
            degree = [row.bit_count() for row in graph.adjacency]
            scores = [degree[u] + degree[v] for u, v in candidates]
            smallest = min(scores)
            edge = rng.choice([e for e, s in zip(candidates, scores, strict=True) if s == smallest])
        else:
            raise ValueError("unknown greedy method")
        if time.perf_counter() >= deadline:
            break
        graph.add(*edge)
    return graph


def cycle_cuts(graph: Graph) -> set[tuple[Edge, ...]]:
    """Enumerate present C4 edge sets; duplicates through opposite pairs removed."""
    cycles = set()
    for u, v in itertools.combinations(range(graph.n), 2):
        for a, b in itertools.combinations(vertices(graph.adjacency[u] & graph.adjacency[v]), 2):
            cycles.add(tuple(sorted(tuple(sorted(e)) for e in [(u, a), (a, v), (v, b), (b, u)])))
    return cycles


def verify_witness(n: int, edge_list: Iterable[Edge], *, explicit_cycles: bool = False) -> dict:
    """Independent of bitsets, updates, legality masks, and the CP-SAT formulation."""
    edge_list = [tuple(edge) for edge in edge_list]
    neighbors = [set() for _ in range(n)]
    if len(set(edge_list)) != len(edge_list):
        raise ValueError("duplicate edges")
    for edge in edge_list:
        if len(edge) != 2:
            raise ValueError("edge must have two vertices")
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int) or not 0 <= u < v < n:
            raise ValueError("invalid vertex order, loop, or endpoint")
        neighbors[u].add(v)
        neighbors[v].add(u)
    if any(len(neighbors[u] & neighbors[v]) > 1 for u, v in itertools.combinations(range(n), 2)):
        raise ValueError("C4 in witness")
    if explicit_cycles:
        # Distinct independent algorithm: all cyclic orders on each 4-subset.
        for a, b, c, d in itertools.combinations(range(n), 4):
            for w, x, y, z in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
                if (
                    x in neighbors[w]
                    and y in neighbors[x]
                    and z in neighbors[y]
                    and w in neighbors[z]
                ):
                    raise ValueError("explicit four-cycle in witness")
    return {"n": n, "edges": len(edge_list), "c4_free": True, "explicit_cycles": explicit_cycles}

"""Orthogonal polarity graphs over finite fields, without external witnesses."""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass

from .graph import Graph, greedy_fill


def prime_power(q: int) -> tuple[int, int]:
    if q < 2:
        raise ValueError("field order must be a prime power >= 2")
    for p in range(2, q + 1):
        if any(p % d == 0 for d in range(2, int(p**0.5) + 1)) or q % p:
            continue
        value, exponent = q, 0
        while value % p == 0:
            value //= p
            exponent += 1
        if value == 1:
            return p, exponent
        break
    raise ValueError("order is not a prime power")


def polynomial_remainder(a: list[int], b: list[int], p: int) -> list[int]:
    a = a.copy()
    while a and not a[-1]:
        a.pop()
    while len(a) >= len(b):
        factor = a[-1] * pow(b[-1], -1, p) % p
        offset = len(a) - len(b)
        for j, value in enumerate(b):
            a[offset + j] = (a[offset + j] - factor * value) % p
        while a and not a[-1]:
            a.pop()
    return a


def irreducible_polynomial(p: int, degree: int) -> list[int]:
    for low in itertools.product(range(p), repeat=degree):
        polynomial = [*low, 1]
        if not polynomial[0]:
            continue
        if all(
            polynomial_remainder(polynomial, [*factor, 1], p)
            for k in range(1, degree // 2 + 1)
            for factor in itertools.product(range(p), repeat=k)
        ):
            return polynomial
    raise AssertionError("no irreducible polynomial")


class FiniteField:
    """Polynomial basis, integer encodings of coefficients in increasing powers."""

    def __init__(self, q: int):
        self.q = q
        self.p, self.degree = prime_power(q)
        self.modulus = irreducible_polynomial(self.p, self.degree) if self.degree > 1 else None
        self.coefficients = [
            [(a // self.p**j) % self.p for j in range(self.degree)] for a in range(q)
        ]
        self.add_table = [[self._add(a, b) for b in range(q)] for a in range(q)]
        self.mul_table = [[self._mul(a, b) for b in range(q)] for a in range(q)]

    def _add(self, a: int, b: int) -> int:
        return sum(
            (x + y) % self.p * self.p**j
            for j, (x, y) in enumerate(zip(self.coefficients[a], self.coefficients[b], strict=True))
        )

    def _mul(self, a: int, b: int) -> int:
        if self.degree == 1:
            return a * b % self.p
        product = [0] * (2 * self.degree - 1)
        for i, x in enumerate(self.coefficients[a]):
            for j, y in enumerate(self.coefficients[b]):
                product[i + j] = (product[i + j] + x * y) % self.p
        reduced = polynomial_remainder(product, self.modulus, self.p)
        return sum(x * self.p**i for i, x in enumerate(reduced))


def polarity_graph(q: int) -> Graph:
    """PG(2,q), normalized representatives, distinct orthogonal pairs (no loops)."""
    field = FiniteField(q)
    points = [(1, a, b) for a in range(q) for b in range(q)]
    points += [(0, 1, a) for a in range(q)] + [(0, 0, 1)]
    graph = Graph.empty(len(points))
    for u, v in itertools.combinations(range(len(points)), 2):
        dot = 0
        for a, b in zip(points[u], points[v], strict=True):
            dot = field.add_table[dot][field.mul_table[a][b]]
        if dot == 0:
            graph.add(u, v)
    return graph


@dataclass
class StartingGraph:
    graph: Graph
    kind: str
    field_order: int | None
    completed_at: float


def starting_pool(n: int, seed: int, *, deadline: float = float("inf")) -> list[StartingGraph]:
    if not 2 <= n <= 133:
        raise ValueError("current construction panel supports orders 2..133")
    rng = random.Random(seed)
    orders = [2, 3, 4, 5, 7, 8, 9, 11]
    above = next(q for q in orders if q * q + q + 1 >= n)
    below = [q for q in orders if q * q + q + 1 < n]
    pool = []
    for q in [above, *below[-1:]]:
        if time.perf_counter() >= deadline:
            break
        source = polarity_graph(q)
        if source.n > n:
            remaining = list(range(source.n))
            while len(remaining) > n:
                mask = sum(1 << u for u in remaining)
                degrees = [(source.adjacency[u] & mask).bit_count() for u in remaining]
                smallest = min(degrees)
                remaining.remove(
                    rng.choice(
                        [u for u, d in zip(remaining, degrees, strict=True) if d == smallest]
                    )
                )
            graph = source.induced(remaining)
            kind = "trim_polarity"
        else:
            graph = Graph(source.adjacency + [0] * (n - source.n))
            kind = "extend_polarity" if source.n < n else "polarity"
        greedy_fill(graph, rng, deadline=deadline)
        if time.perf_counter() < deadline:
            pool.append(StartingGraph(graph, kind, q, time.perf_counter()))
    for method in ["min_degree", "uniform"]:
        if time.perf_counter() >= deadline:
            break
        graph = greedy_fill(Graph.empty(n), rng, deadline=deadline, method=method)
        if time.perf_counter() < deadline:
            pool.append(StartingGraph(graph, method, None, time.perf_counter()))
    # If initialization exceeds its budget, a verified empty incumbent remains available.
    return pool

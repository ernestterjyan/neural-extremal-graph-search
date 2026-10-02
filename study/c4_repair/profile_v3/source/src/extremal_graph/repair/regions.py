"""Shared candidate generation for adaptive, MLP, and GNN neighborhood selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .graph import Graph, vertices


@dataclass(frozen=True)
class Region:
    vertices: tuple[int, ...]
    family: str


def candidate_regions(
    graph: Graph, rng: random.Random, *, sizes: tuple[int, ...] = (3, 5, 7), per_family: int = 4
) -> list[Region]:
    if per_family < 1 or not sizes or any(k < 1 or k > graph.n for k in sizes):
        raise ValueError("invalid candidate pool settings")
    regions = []
    seen = set()
    missing = [(u, v) for u in range(graph.n) for v in range(u + 1, graph.n) if not graph.has(u, v)]

    def emit(selected: list[int], family: str, k: int) -> None:
        selected = list(dict.fromkeys(selected))
        if len(selected) < k:
            selected += rng.sample(
                [u for u in range(graph.n) if u not in selected], k - len(selected)
            )
        ordered = tuple(sorted(selected[:k]))
        if ordered not in seen:
            regions.append(Region(ordered, family))
            seen.add(ordered)

    for k in sizes:
        for _ in range(per_family):
            emit(rng.sample(range(graph.n), k), "random", k)
            root = rng.randrange(graph.n)
            neighbors = list(vertices(graph.adjacency[root]))
            rng.shuffle(neighbors)
            emit([root, *neighbors], "neighborhood", k)
            if missing:
                u, v = rng.choice(missing)
                paths = [
                    (a, b)
                    for a in vertices(graph.adjacency[u])
                    for b in vertices(graph.adjacency[a] & graph.adjacency[v])
                ]
                if paths:
                    a, b = rng.choice(paths)
                    selected = [u, a, b, v]
                    rng.shuffle(selected)
                    emit(selected, "blocking_path", k)
    return regions

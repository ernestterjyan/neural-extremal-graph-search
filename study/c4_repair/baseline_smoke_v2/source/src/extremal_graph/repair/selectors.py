"""Permutation-equivariant GNN and engineered-feature MLP region rankers."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .graph import Graph, vertices
from .regions import Region

NODE_FEATURES = 5
REGION_FEATURES = 20


def features(
    graph: Graph, regions: list[Region]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = graph.n
    adjacency = np.zeros((n, n), dtype=np.float32)
    degree = np.array([row.bit_count() for row in graph.adjacency], dtype=np.float32)
    triangles = np.zeros(n, dtype=np.float32)
    reach = np.zeros(n, dtype=np.float32)
    neighbor_degree = np.zeros(n, dtype=np.float32)
    legal_incidence = np.zeros(n, dtype=np.float32)
    for u, v in graph.legal_edges():
        legal_incidence[u] += 1
        legal_incidence[v] += 1
    for u, row in enumerate(graph.adjacency):
        neighbors = list(vertices(row))
        adjacency[u, neighbors] = 1
        two_hop = 0
        for v in neighbors:
            two_hop |= graph.adjacency[v]
            triangles[u] += (row & graph.adjacency[v]).bit_count() / 2
        reach[u] = two_hop.bit_count() / max(1, n)
        neighbor_degree[u] = degree[neighbors].mean() if neighbors else 0
    scale = math.sqrt(n)
    node = np.column_stack(
        [
            degree / scale,
            neighbor_degree / scale,
            2 * triangles / np.maximum(1, degree * (degree - 1)),
            reach,
            legal_incidence / n,
        ]
    ).astype(np.float32)
    region_features = []
    for region in regions:
        selected = list(region.vertices)
        r = len(selected)
        mask = sum(1 << u for u in selected)
        internal = sum((graph.adjacency[u] & mask).bit_count() for u in selected) / 2
        boundary = float(degree[selected].sum()) - 2 * internal
        ds = degree[selected] / scale
        ts = node[selected, 2]
        rs = reach[selected]
        region_features.append(
            [
                n / 100,
                r / n,
                graph.m / (n * scale),
                ds.mean(),
                ds.min(),
                ds.max(),
                ds.std(),
                internal / max(1, r * (r - 1) / 2),
                boundary / max(1, r * (n - r)),
                ts.mean(),
                ts.max(),
                rs.mean(),
                rs.min(),
                rs.max(),
                node[selected, 1].mean(),
                node[selected, 4].mean(),
                r / 10,
                *[
                    float(region.family == family)
                    for family in ["random", "neighborhood", "blocking_path"]
                ],
            ]
        )
    return (
        torch.from_numpy(adjacency),
        torch.from_numpy(node),
        torch.tensor(region_features, dtype=torch.float32),
    )


class RegionMLP(nn.Module):
    family = "mlp"
    architecture = "region-mlp-v1"

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(REGION_FEATURES, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, adjacency, node, region_features, regions):
        return self.network(region_features).squeeze(-1)


class RegionGNN(nn.Module):
    family = "gnn"
    architecture = "marked-region-gnn-v1"

    def __init__(self, hidden: int = 32, layers: int = 3):
        super().__init__()
        self.input = nn.Linear(NODE_FEATURES + 1, hidden)
        self.layers = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.score = nn.Sequential(
            nn.Linear(3 * hidden + REGION_FEATURES, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, adjacency, node, region_features, regions):
        if not regions:
            return node.new_empty((0,))
        count, n = len(regions), node.shape[0]
        membership = node.new_zeros((count, n, 1))
        for index, region in enumerate(regions):
            membership[index, list(region.vertices), 0] = 1
        conditioned = torch.cat([node.expand(count, -1, -1), membership], dim=2)
        hidden = torch.nn.functional.silu(self.input(conditioned))
        divisor = adjacency.sum(dim=1).clamp_min(1).view(1, n, 1)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            neighbor = adjacency @ hidden / divisor
            hidden = norm(
                hidden + torch.nn.functional.silu(layer(torch.cat([hidden, neighbor], dim=2)))
            )
        selected_mean = (hidden * membership).sum(dim=1) / membership.sum(dim=1).clamp_min(1)
        selected_max = hidden.masked_fill(~membership.bool(), -torch.inf).amax(dim=1)
        pooled = torch.cat([selected_mean, selected_max, hidden.mean(dim=1)], dim=1)
        return self.score(torch.cat([pooled, region_features], dim=1)).squeeze(-1)


def make_model(family: str) -> nn.Module:
    if family == "gnn":
        return RegionGNN()
    if family == "mlp":
        return RegionMLP()
    raise ValueError("unknown learned family")


def model_selector(model: nn.Module):
    model.eval()

    def select(graph: Graph, regions: list[Region]) -> list[float]:
        with torch.inference_mode():
            adjacency, node, engineered = features(graph, regions)
            return model(adjacency, node, engineered, regions).tolist()

    return select

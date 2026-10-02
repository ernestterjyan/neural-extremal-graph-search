"""Region-conditioned GNN staged for integration after the frozen live gate.

All candidates get the same graph/node/engineered features. A candidate membership
bit lets message passing describe joint region geometry before pooling. Inference
must remain within the common timed search budget.
"""

from __future__ import annotations

import torch
from torch import nn

from extremal_graph.repair.selectors import NODE_FEATURES, REGION_FEATURES


class MarkedRegionGNN(nn.Module):
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

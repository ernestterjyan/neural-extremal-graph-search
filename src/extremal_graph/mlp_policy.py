"""Fixed-capacity, non-equivariant MLP control policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .features import GraphTensorBatch


class FixedSizeMLPPolicy(nn.Module):
    """Score legal edges from a fixed padded graph representation.

    Unlike :class:`~extremal_graph.policy.EdgePolicy`, this control assigns
    distinct input weights to fixed vertex positions. It can therefore run only
    up to ``max_nodes`` and is deliberately not permutation equivariant.
    Candidate endpoint encoding remains symmetric because edges are undirected.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = 4,
        candidate_feature_dim: int = 5,
        hidden_dim: int = 64,
        max_nodes: int = 24,
    ) -> None:
        super().__init__()
        if node_feature_dim < 1 or candidate_feature_dim < 0 or hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        self.node_feature_dim = node_feature_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        upper = torch.triu_indices(max_nodes, max_nodes, offset=1)
        self.register_buffer("upper_rows", upper[0], persistent=False)
        self.register_buffer("upper_columns", upper[1], persistent=False)
        graph_input_dim = (
            max_nodes * (max_nodes - 1) // 2 + max_nodes * node_feature_dim + max_nodes
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        scorer_input_dim = hidden_dim + max_nodes + candidate_feature_dim
        self.edge_scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _graph_input(self, batch: GraphTensorBatch) -> torch.Tensor:
        node_count = batch.adjacency.shape[1]
        if node_count > self.max_nodes:
            raise ValueError(
                f"batch has {node_count} nodes but fixed-size MLP capacity is {self.max_nodes}"
            )
        padding = self.max_nodes - node_count
        adjacency = F.pad(batch.adjacency, (0, padding, 0, padding))
        node_features = F.pad(batch.node_features, (0, 0, 0, padding))
        node_mask = F.pad(batch.node_mask, (0, padding)).to(batch.node_features.dtype)
        upper_adjacency = adjacency[:, self.upper_rows, self.upper_columns]
        return torch.cat(
            [upper_adjacency, node_features.flatten(start_dim=1), node_mask],
            dim=1,
        )

    def forward(self, batch: GraphTensorBatch) -> torch.Tensor:
        graph_embedding = self.graph_encoder(self._graph_input(batch))
        candidate_count = batch.candidates.shape[1]
        graph_by_candidate = graph_embedding.unsqueeze(1).expand(-1, candidate_count, -1)
        endpoint_positions = F.one_hot(
            batch.candidates,
            num_classes=self.max_nodes,
        ).to(batch.node_features.dtype)
        symmetric_positions = endpoint_positions.sum(dim=2)
        scorer_input = torch.cat(
            [graph_by_candidate, symmetric_positions, batch.candidate_features],
            dim=-1,
        )
        logits = self.edge_scorer(scorer_input).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)

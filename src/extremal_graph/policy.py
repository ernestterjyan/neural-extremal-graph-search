"""Custom dense permutation-equivariant edge policy."""

from __future__ import annotations

import torch
from torch import nn

from .features import GraphTensorBatch


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbour_mean = adjacency.bmm(hidden) / degree
        update = self.activation(self.update(torch.cat([hidden, neighbour_mean], dim=-1)))
        result = self.normalization(hidden + update)
        return result * node_mask.unsqueeze(-1)


class EdgePolicy(nn.Module):
    """Score a variable legal-edge set with endpoint-symmetric features."""

    def __init__(
        self,
        *,
        node_feature_dim: int = 4,
        candidate_feature_dim: int = 5,
        hidden_dim: int = 64,
        message_passing_layers: int = 3,
    ) -> None:
        super().__init__()
        if node_feature_dim < 1 or candidate_feature_dim < 0 or hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if message_passing_layers < 1:
            raise ValueError("at least one message-passing layer is required")
        self.node_feature_dim = node_feature_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.message_passing_layers = message_passing_layers
        self.input_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim) for _ in range(message_passing_layers)
        )
        scorer_input = 3 * hidden_dim + candidate_feature_dim
        self.edge_scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def model_config(self) -> dict[str, int]:
        return {
            "node_feature_dim": self.node_feature_dim,
            "candidate_feature_dim": self.candidate_feature_dim,
            "hidden_dim": self.hidden_dim,
            "message_passing_layers": self.message_passing_layers,
        }

    def node_embeddings(self, batch: GraphTensorBatch) -> torch.Tensor:
        hidden = self.input_projection(batch.node_features)
        hidden = hidden * batch.node_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, batch.adjacency, batch.node_mask)
        return hidden

    def forward(self, batch: GraphTensorBatch) -> torch.Tensor:
        hidden = self.node_embeddings(batch)
        embedding_dim = hidden.shape[-1]
        u_indices = batch.candidates[:, :, 0].unsqueeze(-1).expand(-1, -1, embedding_dim)
        v_indices = batch.candidates[:, :, 1].unsqueeze(-1).expand(-1, -1, embedding_dim)
        h_u = hidden.gather(1, u_indices)
        h_v = hidden.gather(1, v_indices)
        symmetric = torch.cat(
            [h_u + h_v, torch.abs(h_u - h_v), h_u * h_v, batch.candidate_features],
            dim=-1,
        )
        logits = self.edge_scorer(symmetric).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)


def sample_actions(
    masked_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one candidate index per active row; return ``-1`` for terminal rows."""
    if masked_logits.ndim != 2:
        raise ValueError("masked_logits must have shape [batch, candidates]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    active = torch.isfinite(masked_logits).any(dim=1)
    safe_logits = masked_logits.detach().to("cpu") / temperature
    safe_logits[~active.to("cpu")] = -torch.inf
    safe_logits[~active.to("cpu"), 0] = 0.0
    probabilities = torch.softmax(safe_logits, dim=1)
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    sampled[~active.to("cpu")] = -1
    return sampled.to(masked_logits.device)

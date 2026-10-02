"""Custom dense permutation-equivariant edge policy."""

from __future__ import annotations

import math

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
        if message_passing_layers < 0:
            raise ValueError("message-passing depth cannot be negative")
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


def action_probabilities(
    masked_logits: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate legal logits; reserve -inf exclusively for padding/terminal masks."""
    if masked_logits.ndim != 2 or masked_logits.shape[1] == 0:
        raise ValueError("masked_logits must have shape [batch, nonempty candidates]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = masked_logits.detach().to("cpu")
    # Compatibility for callers without an explicit mask: -inf denotes masking.
    # Rollouts always supply the true mask, allowing detection of illegal -inf.
    mask = ~torch.isneginf(logits) if candidate_mask is None else candidate_mask.to("cpu")
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean and match logits")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise FloatingPointError("policy produced NaN or positive-infinite logits")
    if not torch.isfinite(logits[mask]).all():
        raise FloatingPointError("policy produced a nonfinite legal-candidate logit")
    if not torch.isneginf(logits[~mask]).all():
        raise ValueError("masked candidates must have negative-infinity logits")
    active = mask.any(dim=1)
    safe = logits.clone()
    safe[~active, 0] = 0.0
    scaled = safe / temperature
    if not torch.isfinite(scaled[mask]).all():
        raise FloatingPointError("temperature scaling overflowed legal logits")
    probabilities = torch.softmax(scaled, dim=1)
    return probabilities, active


def sample_actions(
    masked_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    generators: list[torch.Generator] | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample valid indices, with independent inverse-CDF streams when supplied.

    The single-generator path preserves historical torch.multinomial sampling.
    One uniform draw per active episode makes episode-v2 streams independent
    of batch order, padding, companion termination, and chunk boundaries.
    """
    probabilities, active = action_probabilities(
        masked_logits, candidate_mask=candidate_mask, temperature=temperature
    )
    if generators is not None:
        if generator is not None or len(generators) != len(probabilities):
            raise ValueError("supply exactly one generator per row, not both generator modes")
        sampled = torch.full((len(probabilities),), -1, dtype=torch.long)
        for row, rng in enumerate(generators):
            if active[row]:
                uniform = torch.rand((), generator=rng, dtype=torch.float64)
                cdf = probabilities[row].double().cumsum(0)
                cdf = cdf / cdf[-1]
                sampled[row] = torch.searchsorted(cdf, uniform, right=True).clamp_max(len(cdf) - 1)
    else:
        sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        sampled[~active] = -1
    return sampled.to(masked_logits.device)

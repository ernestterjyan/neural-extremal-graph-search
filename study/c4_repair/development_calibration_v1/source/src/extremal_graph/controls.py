"""Small learned controls that separate supplied statistics from message passing."""

from torch import nn

from .features import GraphTensorBatch


class CandidatePolicy(nn.Module):
    def __init__(self, candidate_feature_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: GraphTensorBatch):
        return (
            self.scorer(batch.candidate_features)
            .squeeze(-1)
            .masked_fill(~batch.candidate_mask, -float("inf"))
        )

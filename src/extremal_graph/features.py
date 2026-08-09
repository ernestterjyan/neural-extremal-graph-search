"""Permutation-equivariant graph and candidate feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch

from .graph import Edge, GraphState
from .turan import turan_edge_count


def legal_edges_from_state(state: GraphState) -> list[Edge]:
    """Compute triangle-free legal actions from an immutable state."""
    neighbours = [set() for _ in range(state.n)]
    existing = set(state.edges)
    for u, v in state.edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    return [
        (u, v)
        for u in range(state.n)
        for v in range(u + 1, state.n)
        if (u, v) not in existing and neighbours[u].isdisjoint(neighbours[v])
    ]


@dataclass(frozen=True, slots=True)
class GraphTensorBatch:
    """A dynamic padded batch consumed by neural edge policies."""

    adjacency: torch.Tensor
    node_features: torch.Tensor
    node_mask: torch.Tensor
    candidates: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.adjacency.shape[0]

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            adjacency=self.adjacency.to(device),
            node_features=self.node_features.to(device),
            node_mask=self.node_mask.to(device),
            candidates=self.candidates.to(device),
            candidate_features=self.candidate_features.to(device),
            candidate_mask=self.candidate_mask.to(device),
        )


def collate_graph_states(
    states: list[GraphState] | tuple[GraphState, ...],
    candidate_lists: list[list[Edge]] | tuple[list[Edge], ...] | None = None,
) -> GraphTensorBatch:
    """Convert immutable states into padded tensors without using vertex IDs."""
    if not states:
        raise ValueError("at least one graph state is required")
    if any(state.r != 2 for state in states):
        raise ValueError("MVP feature construction supports only r = 2")

    candidates_by_graph = (
        [legal_edges_from_state(state) for state in states]
        if candidate_lists is None
        else list(candidate_lists)
    )
    if len(candidates_by_graph) != len(states):
        raise ValueError("candidate_lists must have one entry per state")

    batch_size = len(states)
    max_nodes = max(state.n for state in states)
    max_candidates = max(1, max(len(values) for values in candidates_by_graph))
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float32)
    node_features = torch.zeros((batch_size, max_nodes, 4), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    candidates = torch.zeros((batch_size, max_candidates, 2), dtype=torch.long)
    candidate_features = torch.zeros((batch_size, max_candidates, 5), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

    for graph_index, (state, legal_edges) in enumerate(
        zip(states, candidates_by_graph, strict=True)
    ):
        n = state.n
        node_mask[graph_index, :n] = True
        neighbours = [set() for _ in range(n)]
        for u, v in state.edges:
            adjacency[graph_index, u, v] = 1.0
            adjacency[graph_index, v, u] = 1.0
            neighbours[u].add(v)
            neighbours[v].add(u)

        degrees = [len(values) for values in neighbours]
        legal_incident = [0] * n
        for u, v in legal_edges:
            legal_incident[u] += 1
            legal_incident[v] += 1

        scale = max(1, n - 1)
        progress = state.edge_count / max(1, turan_edge_count(n, state.r))
        for vertex in range(n):
            node_features[graph_index, vertex] = torch.tensor(
                [
                    degrees[vertex] / scale,
                    legal_incident[vertex] / scale,
                    progress,
                    1.0 / state.r,
                ]
            )

        for candidate_index, (u, v) in enumerate(legal_edges):
            candidates[graph_index, candidate_index] = torch.tensor([u, v])
            candidate_mask[graph_index, candidate_index] = True
            candidate_features[graph_index, candidate_index] = torch.tensor(
                [
                    (degrees[u] + degrees[v]) / (2 * scale),
                    abs(degrees[u] - degrees[v]) / scale,
                    (legal_incident[u] + legal_incident[v]) / (2 * scale),
                    abs(legal_incident[u] - legal_incident[v]) / scale,
                    len(neighbours[u] & neighbours[v]) / max(1, n - 2),
                ]
            )

    return GraphTensorBatch(
        adjacency=adjacency,
        node_features=node_features,
        node_mask=node_mask,
        candidates=candidates,
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
    )


def action_indices(candidate_lists: list[list[Edge]], actions: list[Edge]) -> torch.Tensor:
    if len(candidate_lists) != len(actions):
        raise ValueError("candidate_lists and actions must have equal lengths")
    indices: list[int] = []
    for candidates, action in zip(candidate_lists, actions, strict=True):
        try:
            indices.append(candidates.index(action))
        except ValueError as error:
            raise ValueError(f"action {action} is not present in its candidate list") from error
    return torch.tensor(indices, dtype=torch.long)

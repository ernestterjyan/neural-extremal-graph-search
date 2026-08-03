import pytest
import torch

from extremal_graph import EdgePolicy, GraphState, sample_actions
from extremal_graph.features import collate_graph_states, legal_edges_from_state


def _probability_map(model: EdgePolicy, state: GraphState) -> dict[tuple[int, int], float]:
    candidates = legal_edges_from_state(state)
    batch = collate_graph_states([state], [candidates])
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    return {edge: float(probabilities[index]) for index, edge in enumerate(candidates)}


def test_policy_shapes_masks_and_terminal_sampling() -> None:
    states = [
        GraphState(n=4, r=2, edges=()),
        GraphState(n=5, r=2, edges=((0, 1), (1, 2))),
    ]
    candidate_lists = [legal_edges_from_state(state) for state in states]
    batch = collate_graph_states(states, candidate_lists)
    model = EdgePolicy(hidden_dim=16, message_passing_layers=2)
    logits = model(batch)
    assert logits.shape == batch.candidate_mask.shape
    assert torch.isfinite(logits[batch.candidate_mask]).all()
    assert torch.isneginf(logits[~batch.candidate_mask]).all()
    probabilities = torch.softmax(logits, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(
        probabilities[~batch.candidate_mask], torch.zeros_like(probabilities[~batch.candidate_mask])
    )

    terminal = GraphState(n=2, r=2, edges=((0, 1),))
    terminal_batch = collate_graph_states([terminal], [[]])
    terminal_logits = model(terminal_batch)
    sampled = sample_actions(terminal_logits, generator=torch.Generator().manual_seed(1))
    assert sampled.tolist() == [-1]


def test_policy_is_equivariant_to_vertex_permutation() -> None:
    torch.manual_seed(5)
    model = EdgePolicy(hidden_dim=24, message_passing_layers=3)
    model.eval()
    state = GraphState(n=6, r=2, edges=((0, 1), (1, 2), (3, 4)))
    original = _probability_map(model, state)

    permutation = [4, 1, 5, 0, 3, 2]
    permuted_edges = tuple(
        sorted(
            (min(permutation[u], permutation[v]), max(permutation[u], permutation[v]))
            for u, v in state.edges
        )
    )
    permuted_state = GraphState(n=6, r=2, edges=permuted_edges)
    permuted = _probability_map(model, permuted_state)
    mapped_back = {
        edge: permuted[
            (
                min(permutation[edge[0]], permutation[edge[1]]),
                max(permutation[edge[0]], permutation[edge[1]]),
            )
        ]
        for edge in original
    }
    for edge in original:
        assert original[edge] == pytest.approx(mapped_back[edge], abs=2e-7)


def test_padding_does_not_change_policy_distribution() -> None:
    torch.manual_seed(11)
    model = EdgePolicy(hidden_dim=16, message_passing_layers=2).eval()
    small = GraphState(n=4, r=2, edges=((0, 1),))
    large = GraphState(n=9, r=2, edges=((0, 1), (2, 3), (4, 5)))
    small_candidates = legal_edges_from_state(small)

    alone = collate_graph_states([small], [small_candidates])
    padded = collate_graph_states([small, large], [small_candidates, legal_edges_from_state(large)])
    with torch.inference_mode():
        alone_probabilities = torch.softmax(model(alone), dim=1)[0]
        padded_probabilities = torch.softmax(model(padded), dim=1)[0, : len(small_candidates)]
    assert torch.allclose(alone_probabilities, padded_probabilities, atol=1e-7)


def test_sampling_is_seeded() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    first = sample_actions(logits, generator=torch.Generator().manual_seed(99))
    second = sample_actions(logits, generator=torch.Generator().manual_seed(99))
    assert torch.equal(first, second)

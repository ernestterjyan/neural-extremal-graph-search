import torch
from torch.nn import functional as F

from experiments.c4_neural_pilot import collate, generate, model_for
from experiments.c4_solver_pilot import edges, legal_edges, verify_graph


def test_c4_neural_generation_is_valid_and_batch_order_independent():
    torch.set_num_threads(1)
    torch.manual_seed(55)
    model = model_for("gnn")
    seeds = [1201, 1202, 1203]
    grouped = generate(model, 8, seeds, trace=True)
    reversed_group = generate(model, 8, list(reversed(seeds)), trace=True)
    singleton = generate(model, 8, seeds[:1], trace=True)
    for index, result in enumerate(grouped):
        verify_graph(8, edges(result["adjacency"]))
        assert result["edge_count"] == reversed_group[2 - index]["edge_count"]
        assert result["adjacency"] == reversed_group[2 - index]["adjacency"]
        for state, candidates, selected in result["trace"]:
            assert list(candidates) == legal_edges(list(state))
            assert 0 <= selected < len(candidates)
    assert grouped[0]["adjacency"] == singleton[0]["adjacency"]


def test_c4_neural_collate_supports_training_gradient():
    torch.set_num_threads(1)
    model = model_for("endpoint")
    adjacency = (0, 0, 0, 0, 0, 0)
    candidates = tuple(legal_edges(list(adjacency)))
    batch = collate([(adjacency, candidates), (adjacency, candidates)])
    logits = model(batch)
    loss = F.cross_entropy(logits, torch.tensor([0, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())

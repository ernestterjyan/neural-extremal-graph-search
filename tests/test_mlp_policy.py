from pathlib import Path

import pytest
import torch

from extremal_graph import (
    EdgePolicy,
    FixedSizeMLPPolicy,
    GraphState,
    run_episode,
    verify_graph,
)
from extremal_graph.features import collate_graph_states, legal_edges_from_state
from extremal_graph.training import load_model_checkpoint, train


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _probability_map(
    model: FixedSizeMLPPolicy,
    state: GraphState,
) -> dict[tuple[int, int], float]:
    candidates = legal_edges_from_state(state)
    batch = collate_graph_states([state], [candidates])
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    return {edge: float(probabilities[index]) for index, edge in enumerate(candidates)}


def test_mlp_shapes_masks_capacity_and_rollout() -> None:
    model = FixedSizeMLPPolicy(hidden_dim=16, max_nodes=8).eval()
    state = GraphState(n=6, r=2, edges=((0, 1), (1, 2)))
    terminal = GraphState(n=2, r=2, edges=((0, 1),))
    candidates = [legal_edges_from_state(state), []]
    batch = collate_graph_states([state, terminal], candidates)

    with torch.inference_mode():
        logits = model(batch)

    assert logits.shape == batch.candidate_mask.shape
    assert torch.isfinite(logits[batch.candidate_mask]).all()
    assert torch.isneginf(logits[~batch.candidate_mask]).all()
    trajectory = run_episode(model, n=8, seed=17)
    assert verify_graph(trajectory.final_state).valid

    oversized = collate_graph_states([GraphState(n=9, r=2)])
    with pytest.raises(ValueError, match="capacity"):
        model(oversized)


def test_mlp_cpu_rollout_is_reproducible() -> None:
    torch.manual_seed(11)
    model = FixedSizeMLPPolicy(hidden_dim=16, max_nodes=8).eval()

    first = run_episode(model, n=8, seed=901)
    second = run_episode(model, n=8, seed=901)

    assert first.actions == second.actions
    assert first.optimality_ratio == second.optimality_ratio


def test_full_mlp_parameter_count_matches_gnn_within_one_percent() -> None:
    gnn_count = _parameter_count(EdgePolicy(hidden_dim=64, message_passing_layers=3))
    mlp_count = _parameter_count(FixedSizeMLPPolicy(hidden_dim=67, max_nodes=24))

    assert abs(gnn_count - mlp_count) / gnn_count < 0.01


def test_mlp_is_deliberately_sensitive_to_vertex_positions() -> None:
    torch.manual_seed(29)
    model = FixedSizeMLPPolicy(hidden_dim=24, max_nodes=8).eval()
    state = GraphState(n=6, r=2, edges=((0, 1), (1, 2), (3, 4)))
    original = _probability_map(model, state)

    permutation = [4, 1, 5, 0, 3, 2]
    permuted_state = GraphState(
        n=6,
        r=2,
        edges=tuple(
            (min(permutation[u], permutation[v]), max(permutation[u], permutation[v]))
            for u, v in state.edges
        ),
    )
    permuted = _probability_map(model, permuted_state)
    mapped = {
        edge: permuted[
            (
                min(permutation[edge[0]], permutation[edge[1]]),
                max(permutation[edge[0]], permutation[edge[1]]),
            )
        ]
        for edge in original
    }
    assert any(abs(original[edge] - mapped[edge]) > 1e-5 for edge in original)


def test_legacy_gnn_checkpoint_still_loads(tmp_path: Path) -> None:
    model = EdgePolicy(hidden_dim=8, message_passing_layers=1)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": model.model_config(),
            "seed": 0,
        },
        checkpoint,
    )

    loaded, payload = load_model_checkpoint(checkpoint)

    assert isinstance(loaded, EdgePolicy)
    assert payload["format_version"] == 1


def test_mlp_smoke_training_and_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "mlp.toml"
    config.write_text(
        f"""
[run]
name = "mlp-test"
output_dir = "{(tmp_path / "runs").as_posix()}"
device = "cpu"
deterministic = true

[environment]
r = 2

[model]
family = "mlp"
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1
max_nodes = 6

[training]
curriculum_sizes = [4]
rollouts_per_iteration = 4
temperature = 1.0
elite_fraction = 0.5
learning_rate = 0.001
weight_decay = 0.0
optimization_epochs = 1
gradient_clip_norm = 1.0
validation_interval = 1
validation_episodes = 2
advancement_threshold = 1.0
advancement_patience = 1
max_iterations_per_stage = 1
seed = 3

[artifacts]
trajectory_sample_count = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    checkpoint = train(config)
    loaded, payload = load_model_checkpoint(checkpoint)

    assert isinstance(loaded, FixedSizeMLPPolicy)
    assert payload["model_family"] == "mlp"
    assert payload["model_config"]["max_nodes"] == 6

from pathlib import Path

import pytest
import torch

import extremal_graph.training as training_module
from extremal_graph import (
    EdgePolicy,
    GraphState,
    Trajectory,
    run_episode,
    turan_edge_count,
    verify_graph,
)
from extremal_graph.training import load_model_checkpoint, train


def _training_config(tmp_path: Path) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(
        f"""
[run]
name = "test"
output_dir = "{(tmp_path / "runs").as_posix()}"
device = "cpu"
deterministic = true

[environment]
r = 2

[model]
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1

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
    return path


def test_neural_rollout_is_reproducible_valid_and_reward_normalized() -> None:
    torch.manual_seed(3)
    model = EdgePolicy(hidden_dim=8, message_passing_layers=1)
    first = run_episode(model, n=8, seed=42)
    second = run_episode(model, n=8, seed=42)
    assert first.actions == second.actions
    assert verify_graph(first.final_state).valid
    assert first.return_value == pytest.approx(
        first.final_state.edge_count / turan_edge_count(8, 2)
    )


def test_training_checkpoint_and_completed_resume(tmp_path) -> None:
    config = _training_config(tmp_path)
    checkpoint = train(config)
    run_dir = checkpoint.parent
    assert checkpoint.exists()
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "config.toml").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "metrics.jsonl").read_text().strip()
    model, payload = load_model_checkpoint(checkpoint)
    assert payload["seed"] == 3
    assert isinstance(model, EdgePolicy)

    resumed = train(config, resume=run_dir / "latest.pt")
    assert resumed == checkpoint
    assert resumed.exists()


def test_training_restores_best_stage_state_before_advancing(tmp_path, monkeypatch) -> None:
    config = tmp_path / "two-stage.toml"
    config.write_text(
        f"""
[run]
name = "restore-test"
output_dir = "{(tmp_path / "runs").as_posix()}"
device = "cpu"
deterministic = true

[environment]
r = 2

[model]
node_feature_dim = 4
candidate_feature_dim = 5
hidden_dim = 8
message_passing_layers = 1

[training]
curriculum_sizes = [4, 5]
rollouts_per_iteration = 2
temperature = 1.0
elite_fraction = 0.5
learning_rate = 0.001
weight_decay = 0.0
optimization_epochs = 1
gradient_clip_norm = 1.0
validation_interval = 1
validation_episodes = 1
advancement_threshold = 1.0
advancement_patience = 3
max_iterations_per_stage = 2
seed = 7

[artifacts]
trajectory_sample_count = 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_collect_rollouts(model, *, sizes, **kwargs):
        del model, kwargs
        return [
            Trajectory(
                n=n,
                r=2,
                states=(),
                actions=(),
                rewards=(),
                final_state=GraphState(n=n, r=2),
                optimality_ratio=0.0,
            )
            for n in sizes
        ]

    weights_before_optimization: list[torch.Tensor] = []
    optimizer_markers_before_optimization: list[float | None] = []

    def fake_optimize(model, optimizer, elite, **kwargs):
        del elite, kwargs
        weights_before_optimization.append(next(model.parameters()).detach().cpu().clone())
        optimizer_markers_before_optimization.append(optimizer.param_groups[0].get("test_marker"))
        value = float(len(weights_before_optimization))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
        optimizer.param_groups[0]["test_marker"] = value
        return value

    validation_scores = iter((0.9, 0.1, 0.8, 0.2))

    def fake_validate(model, *, sizes, **kwargs):
        del model, kwargs
        score = next(validation_scores)
        return {n: score for n in sizes}

    monkeypatch.setattr(training_module, "_collect_rollouts", fake_collect_rollouts)
    monkeypatch.setattr(training_module, "optimize_on_elite", fake_optimize)
    monkeypatch.setattr(training_module, "_validate", fake_validate)

    checkpoint = train(config)

    assert len(weights_before_optimization) == 4
    assert torch.equal(
        weights_before_optimization[2], torch.ones_like(weights_before_optimization[2])
    )
    assert optimizer_markers_before_optimization[2] == 1.0
    latest = torch.load(checkpoint.parent / "latest.pt", map_location="cpu", weights_only=False)
    final_weight = next(iter(latest["model_state"].values()))
    assert torch.equal(final_weight, torch.full_like(final_weight, 3.0))
    assert latest["optimizer_state"]["param_groups"][0]["test_marker"] == 3.0

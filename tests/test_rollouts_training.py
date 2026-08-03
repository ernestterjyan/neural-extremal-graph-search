from pathlib import Path

import pytest
import torch

from extremal_graph import EdgePolicy, run_episode, turan_edge_count, verify_graph
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

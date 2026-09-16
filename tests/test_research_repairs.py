import itertools
import json
import random
from pathlib import Path

import pandas as pd
import pytest
import torch

from extremal_graph.baselines import LookaheadPolicy
from extremal_graph.diagnostics import balanced_completion, preserving_actions
from extremal_graph.env import GraphConstructionEnv
from extremal_graph.features import legal_edges_from_state
from extremal_graph.graph import GraphState
from extremal_graph.policy import sample_actions
from extremal_graph.rollouts import run_episode, run_episode_batch
from extremal_graph.training import load_model_checkpoint, train

from .test_rollouts_training import _training_config


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_legal_logits_rejected(bad):
    for logits in [torch.tensor([[bad, bad]]), torch.tensor([[0.0, bad]])]:
        with pytest.raises(FloatingPointError):
            sample_actions(logits, candidate_mask=torch.ones_like(logits, dtype=torch.bool))


def test_terminal_mask_and_padding():
    logits = torch.tensor([[0.0, -torch.inf], [-torch.inf, -torch.inf]])
    mask = torch.tensor([[True, False], [False, False]])
    assert sample_actions(logits, candidate_mask=mask).tolist() == [0, -1]
    with pytest.raises(FloatingPointError):
        sample_actions(torch.tensor([[float("nan")]]))


def test_nan_policy_fails_and_restores_mode():
    class Bad(torch.nn.Module):
        def forward(self, batch):
            return torch.full_like(batch.candidate_features[:, :, 0], float("nan"))

    policy = Bad().train()
    with pytest.raises(FloatingPointError):
        run_episode(policy, n=6, seed=0)
    assert policy.training


@pytest.mark.parametrize("family", ["gnn", "mlp"])
def test_episode_streams_batch_order_and_chunks(family):
    filename = "mvp-seed-0.pt" if family == "gnn" else "mvp-mlp-seed-0.pt"
    model, _ = load_model_checkpoint(Path("artifacts/checkpoints") / filename)
    seeds = [1, 2, 3, 9, 45]
    alone = {s: run_episode(model, n=12, seed=s).actions for s in seeds}
    for order in [seeds, list(reversed(seeds)), seeds[:2], seeds[2:]]:
        group = run_episode_batch(model, n=12, seeds=order)
        assert all(alone[s] == t.actions for s, t in zip(order, group, strict=True))
    # This pair collided in legacy weighted-sum seeding.
    one = run_episode_batch(model, n=8, seeds=[1, 2])
    two = run_episode_batch(model, n=8, seeds=[3, 1])
    assert [t.actions for t in one] != [t.actions for t in two]


def test_legacy_sampling_replays_archived_batch():
    model, _ = load_model_checkpoint("artifacts/checkpoints/mvp-seed-0.pt")
    seeds = [1_000_000 + 8 * 10_000 + i for i in range(64)]
    outputs = run_episode_batch(model, n=8, seeds=seeds, sampling_protocol="legacy-batch-v1")
    frame = pd.read_csv("results/mvp/evaluation.csv")
    expected = frame[(frame.method == "gnn") & (frame.n == 8) & (frame.seed == 0)].head(64)
    assert [t.final_state.edge_count for t in outputs] == list(expected.edge_count)


def _brute_balance(state):
    for subset in itertools.combinations(range(state.n), state.n // 2):
        side = set(subset)
        if all((u in side) != (v in side) for u, v in state.edges):
            return True
    return False


@pytest.mark.parametrize("seed", range(12))
def test_completion_and_lookahead_against_independent_enumeration(seed):
    rng = random.Random(seed)
    n = 3 + seed % 6
    env = GraphConstructionEnv(n)
    for _ in range(rng.randrange(1, n + 1)):
        legal = env.legal_edges()
        if not legal:
            break
        env.step(rng.choice(legal))
    state = env.state
    legal = env.legal_edges()
    assert balanced_completion(state)[0] == _brute_balance(state)
    labels = preserving_actions(state, legal)
    losses = LookaheadPolicy.losses(state, legal)
    for edge, good, loss in zip(legal, labels, losses, strict=True):
        new = GraphState(n, 2, state.edges + (edge,))
        assert good == _brute_balance(new)
        assert loss == len(legal) - len(legal_edges_from_state(new))


def _resume_config(folder):
    folder.mkdir()
    p = _training_config(folder)
    s = p.read_text().replace("curriculum_sizes = [4]", "curriculum_sizes = [4, 5]")
    s = s.replace("advancement_patience = 1", "advancement_patience = 10")
    s = s.replace("max_iterations_per_stage = 1", "max_iterations_per_stage = 3")
    p.write_text(s)
    return p


def test_actual_interrupted_resume_and_rejected_override(tmp_path, monkeypatch):
    import extremal_graph.training as module

    baseline = train(_resume_config(tmp_path / "uninterrupted"))
    config = _resume_config(tmp_path / "interrupted")
    original = module.atomic_torch_save

    def interrupted(value, path):
        original(value, path)
        if path.name == "latest.pt" and value["global_step"] == 2:
            raise RuntimeError("simulated interruption after durable checkpoint")

    monkeypatch.setattr(module, "atomic_torch_save", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        train(config)
    monkeypatch.setattr(module, "atomic_torch_save", original)
    run = config.parent / "runs/test-seed-3"
    originals = {
        name: (run / name).read_bytes()
        for name in ["metadata.json", "config.toml", "metrics.jsonl"]
    }
    changed = config.parent / "changed.toml"
    changed.write_text(config.read_text().replace("learning_rate = 0.001", "learning_rate = 0.009"))
    with pytest.raises(ValueError, match="contract"):
        train(changed, resume=run / "latest.pt")
    assert all((run / name).read_bytes() == contents for name, contents in originals.items())
    resumed = train(config, resume=run / "latest.pt")
    a = torch.load(baseline, weights_only=False)
    b = torch.load(resumed, weights_only=False)
    assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
    assert a["validation_score"] == b["validation_score"]
    assert (run / "metadata.json").read_bytes() == originals["metadata.json"]
    assert len((run / "resume_events.jsonl").read_text().splitlines()) == 1
    assert json.loads((run / "completion.json").read_text())["global_steps"] == 6

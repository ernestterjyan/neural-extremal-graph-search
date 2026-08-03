"""Reproducible policy and baseline evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .baselines import LeastDegreePolicy, UniformRandomPolicy, run_baseline_episode
from .config import EvaluationConfig, load_evaluation_config
from .graph import GraphState, Trajectory
from .rollouts import run_episode_batch
from .training import build_model, load_model_checkpoint
from .turan import construct_turan, turan_edge_count
from .utils import resolve_device, set_global_seed
from .verification import verify_graph


def _episode_seed(seed: int, n: int, episode: int) -> int:
    return (seed + 1) * 1_000_000 + n * 10_000 + episode


def _checkpoint_map(config: EvaluationConfig) -> dict[int, Path]:
    paths = sorted(Path(config.run_dir).glob(config.checkpoint_glob))
    result: dict[int, Path] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_config") != asdict(config.model):
            raise ValueError(f"checkpoint model configuration does not match evaluation: {path}")
        checkpoint_seed = int(payload["seed"])
        if checkpoint_seed in result:
            raise ValueError(f"multiple checkpoints found for seed {checkpoint_seed}")
        result[checkpoint_seed] = path
    return result


def _row(
    *,
    method: str,
    trajectory: Trajectory | None,
    final_state: GraphState,
    n: int,
    seed: int,
    episode: int,
    elapsed: float,
    checkpoint: str | None,
) -> dict[str, Any]:
    optimum = turan_edge_count(n, 2)
    report = verify_graph(final_state, r=2, require_maximal=True)
    exact = report.constraint_free and final_state.edge_count == optimum
    return {
        "method": method,
        "n": n,
        "seed": seed,
        "episode": episode,
        "edge_count": final_state.edge_count,
        "optimal_edge_count": optimum,
        "optimality_ratio": final_state.edge_count / optimum,
        "absolute_gap": optimum - final_state.edge_count,
        "exact_optimum": int(exact),
        # By Turán's theorem, every triangle-free graph attaining ex(n, K3)
        # is isomorphic to the balanced complete bipartite graph.
        "turan_isomorphic": int(exact),
        "constraint_violations": len(report.forbidden_cliques),
        "terminal_maximal": int(report.maximal),
        "episode_length": 0 if trajectory is None else len(trajectory.actions),
        "inference_time_seconds": elapsed,
        "checkpoint": checkpoint,
    }


def _evaluate_neural(
    *,
    method: str,
    model: torch.nn.Module,
    checkpoint: str | None,
    n: int,
    seed: int,
    episodes: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = min(64, episodes)
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        seeds = [_episode_seed(seed, n, start + offset) for offset in range(count)]
        started = time.perf_counter()
        trajectories = run_episode_batch(
            model, n=n, seeds=seeds, r=2, temperature=1.0, device=device
        )
        elapsed = (time.perf_counter() - started) / count
        for offset, trajectory in enumerate(trajectories):
            rows.append(
                _row(
                    method=method,
                    trajectory=trajectory,
                    final_state=trajectory.final_state,
                    n=n,
                    seed=seed,
                    episode=start + offset,
                    elapsed=elapsed,
                    checkpoint=checkpoint,
                )
            )
    return rows


def evaluate(config_path: str | Path) -> Path:
    config = load_evaluation_config(config_path)
    device = resolve_device(config.device)
    checkpoints = _checkpoint_map(config)
    if "gnn" in config.methods:
        missing = set(config.seeds) - checkpoints.keys()
        if missing:
            raise FileNotFoundError(f"missing trained checkpoints for seeds: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        trained_model = None
        checkpoint_path = checkpoints.get(seed)
        if "gnn" in config.methods and checkpoint_path is not None:
            trained_model, _ = load_model_checkpoint(checkpoint_path, device=device)

        untrained_model = None
        if "untrained_gnn" in config.methods:
            set_global_seed(seed, deterministic=True)
            untrained_model = build_model(config.model).to(device)
            untrained_model.eval()

        for n in config.sizes:
            for method in config.methods:
                if method == "gnn":
                    if trained_model is None or checkpoint_path is None:
                        raise RuntimeError(f"trained model for seed {seed} was not loaded")
                    rows.extend(
                        _evaluate_neural(
                            method=method,
                            model=trained_model,
                            checkpoint=str(checkpoint_path),
                            n=n,
                            seed=seed,
                            episodes=config.episodes_per_method,
                            device=device,
                        )
                    )
                elif method == "untrained_gnn":
                    if untrained_model is None:
                        raise RuntimeError("untrained model was not initialized")
                    rows.extend(
                        _evaluate_neural(
                            method=method,
                            model=untrained_model,
                            checkpoint=None,
                            n=n,
                            seed=seed,
                            episodes=config.episodes_per_method,
                            device=device,
                        )
                    )
                elif method in {"random", "least_degree"}:
                    policy = UniformRandomPolicy() if method == "random" else LeastDegreePolicy()
                    for episode in range(config.episodes_per_method):
                        started = time.perf_counter()
                        trajectory = run_baseline_episode(
                            n,
                            policy,
                            seed=_episode_seed(seed, n, episode),
                            r=2,
                        )
                        rows.append(
                            _row(
                                method=method,
                                trajectory=trajectory,
                                final_state=trajectory.final_state,
                                n=n,
                                seed=seed,
                                episode=episode,
                                elapsed=time.perf_counter() - started,
                                checkpoint=None,
                            )
                        )
                elif method == "turan_oracle":
                    for episode in range(config.episodes_per_method):
                        started = time.perf_counter()
                        graph = construct_turan(n, 2)
                        rows.append(
                            _row(
                                method=method,
                                trajectory=None,
                                final_state=graph,
                                n=n,
                                seed=seed,
                                episode=episode,
                                elapsed=time.perf_counter() - started,
                                checkpoint=None,
                            )
                        )

    output = Path(config.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    if int(frame["constraint_violations"].sum()) != 0:
        raise RuntimeError("evaluation produced constraint violations")
    if not bool(frame["terminal_maximal"].all()):
        raise RuntimeError("evaluation produced non-maximal terminal graphs")
    return output

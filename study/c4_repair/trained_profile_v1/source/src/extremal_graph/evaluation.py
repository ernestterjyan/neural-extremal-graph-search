"""Reproducible policy and baseline evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .baselines import LeastDegreePolicy, UniformRandomPolicy, run_baseline_episode
from .config import ModelConfig, load_evaluation_config
from .graph import GraphState, Trajectory
from .rollouts import run_episode_batch
from .training import build_model, load_model_checkpoint
from .turan import construct_turan, turan_edge_count
from .utils import resolve_device, set_global_seed
from .verification import verify_graph


def _episode_seed(seed: int, n: int, episode: int) -> int:
    return (seed + 1) * 1_000_000 + n * 10_000 + episode


def _checkpoint_map(
    *, run_dir: str, checkpoint_glob: str, model_config: ModelConfig
) -> dict[int, Path]:
    paths = sorted(Path(run_dir).glob(checkpoint_glob))
    result: dict[int, Path] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_config = ModelConfig(**payload["model_config"])
        if payload.get("model_family", checkpoint_config.family) != checkpoint_config.family:
            raise ValueError(f"checkpoint model family is internally inconsistent: {path}")
        if checkpoint_config != model_config:
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
    parameter_count: int,
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
        "parameter_count": parameter_count,
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
                    parameter_count=parameter_count,
                )
            )
    return rows


def evaluate(config_path: str | Path) -> Path:
    config = load_evaluation_config(config_path)
    device = resolve_device(config.device)
    checkpoints = _checkpoint_map(
        run_dir=config.run_dir,
        checkpoint_glob=config.checkpoint_glob,
        model_config=config.model,
    )
    mlp_checkpoints = (
        _checkpoint_map(
            run_dir=config.run_dir,
            checkpoint_glob=config.mlp_checkpoint_glob,
            model_config=config.mlp_model,
        )
        if config.mlp_checkpoint_glob is not None and config.mlp_model is not None
        else {}
    )
    if "gnn" in config.methods:
        missing = set(config.seeds) - checkpoints.keys()
        if missing:
            raise FileNotFoundError(f"missing trained checkpoints for seeds: {sorted(missing)}")
    if "mlp" in config.methods:
        missing = set(config.seeds) - mlp_checkpoints.keys()
        if missing:
            raise FileNotFoundError(f"missing trained MLP checkpoints for seeds: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        trained_model = None
        checkpoint_path = checkpoints.get(seed)
        if "gnn" in config.methods and checkpoint_path is not None:
            trained_model, _ = load_model_checkpoint(checkpoint_path, device=device)

        trained_mlp = None
        mlp_checkpoint_path = mlp_checkpoints.get(seed)
        if "mlp" in config.methods and mlp_checkpoint_path is not None:
            trained_mlp, _ = load_model_checkpoint(mlp_checkpoint_path, device=device)

        untrained_model = None
        if "untrained_gnn" in config.methods:
            set_global_seed(seed, deterministic=True)
            untrained_model = build_model(config.model).to(device)
            untrained_model.eval()

        untrained_mlp = None
        if "untrained_mlp" in config.methods:
            if config.mlp_model is None:
                raise RuntimeError("MLP model configuration was not loaded")
            set_global_seed(seed, deterministic=True)
            untrained_mlp = build_model(config.mlp_model).to(device)
            untrained_mlp.eval()

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
                elif method == "mlp":
                    if trained_mlp is None or mlp_checkpoint_path is None:
                        raise RuntimeError(f"trained MLP for seed {seed} was not loaded")
                    rows.extend(
                        _evaluate_neural(
                            method=method,
                            model=trained_mlp,
                            checkpoint=str(mlp_checkpoint_path),
                            n=n,
                            seed=seed,
                            episodes=config.episodes_per_method,
                            device=device,
                        )
                    )
                elif method == "untrained_mlp":
                    if untrained_mlp is None:
                        raise RuntimeError("untrained MLP was not initialized")
                    rows.extend(
                        _evaluate_neural(
                            method=method,
                            model=untrained_mlp,
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
                                parameter_count=0,
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
                                parameter_count=0,
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

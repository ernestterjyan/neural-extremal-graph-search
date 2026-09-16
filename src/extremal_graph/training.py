"""Elite-trajectory cross-entropy training with a size curriculum."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import FullTrainingConfig, ModelConfig, load_training_config
from .controls import CandidatePolicy
from .features import action_indices, collate_graph_states, legal_edges_from_state
from .graph import Edge, GraphState, Trajectory
from .mlp_policy import FixedSizeMLPPolicy
from .policy import EdgePolicy
from .rollouts import run_episode_batch
from .serialization import graph_to_dict
from .utils import (
    append_jsonl,
    atomic_torch_save,
    capture_random_state,
    environment_metadata,
    resolve_device,
    restore_random_state,
    save_source_snapshot,
    set_global_seed,
    sha256_file,
    write_json,
)


def build_model(config: ModelConfig) -> nn.Module:
    if config.family == "gnn":
        return EdgePolicy(
            node_feature_dim=config.node_feature_dim,
            candidate_feature_dim=config.candidate_feature_dim,
            hidden_dim=config.hidden_dim,
            message_passing_layers=config.message_passing_layers,
        )
    if config.family == "candidate":
        return CandidatePolicy(config.candidate_feature_dim, config.hidden_dim)
    if config.family == "endpoint":
        return EdgePolicy(
            node_feature_dim=config.node_feature_dim,
            candidate_feature_dim=config.candidate_feature_dim,
            hidden_dim=config.hidden_dim,
            message_passing_layers=0,
        )
    if config.family == "mlp":
        return FixedSizeMLPPolicy(
            node_feature_dim=config.node_feature_dim,
            candidate_feature_dim=config.candidate_feature_dim,
            hidden_dim=config.hidden_dim,
            max_nodes=config.max_nodes,
        )
    raise ValueError(f"unsupported model family: {config.family!r}")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _write_resolved_toml(path: Path, config: FullTrainingConfig) -> None:
    sections: list[tuple[str, dict[str, Any]]] = [
        ("run", asdict(config.run)),
        ("environment", {"r": config.r}),
        ("model", asdict(config.model)),
        ("training", asdict(config.training)),
        ("artifacts", asdict(config.artifacts)),
    ]
    lines: list[str] = []
    for section, values in sections:
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _rollout_seed(seed: int, global_step: int, n: int, episode: int) -> int:
    return (seed + 1) * 1_000_000_000 + global_step * 100_000 + n * 1_000 + episode


def _collect_rollouts(
    model: nn.Module,
    *,
    sizes: tuple[int, ...],
    count: int,
    seed: int,
    global_step: int,
    r: int,
    temperature: float,
    device: torch.device,
    sampling_protocol: str = "episode-v2",
) -> list[Trajectory]:
    quotient, remainder = divmod(count, len(sizes))
    trajectories: list[Trajectory] = []
    for size_index, n in enumerate(sizes):
        size_count = quotient + (1 if size_index < remainder else 0)
        seeds = [_rollout_seed(seed, global_step, n, episode) for episode in range(size_count)]
        trajectories.extend(
            run_episode_batch(
                model,
                n=n,
                seeds=seeds,
                r=r,
                temperature=temperature,
                device=device,
                sampling_protocol=sampling_protocol,
            )
        )
    return trajectories


def select_elite_trajectories(
    trajectories: list[Trajectory],
    elite_fraction: float,
    rng: random.Random,
) -> list[Trajectory]:
    """Select a seeded top fraction independently within every graph size."""
    grouped: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.n, []).append(trajectory)

    elite: list[Trajectory] = []
    for n in sorted(grouped):
        group = grouped[n][:]
        rng.shuffle(group)
        group.sort(key=lambda item: item.optimality_ratio, reverse=True)
        keep = max(1, math.ceil(len(group) * elite_fraction))
        elite.extend(group[:keep])
    return elite


def _training_pairs(trajectories: list[Trajectory]) -> list[tuple[GraphState, Edge]]:
    return [
        (state, action)
        for trajectory in trajectories
        for state, action in zip(trajectory.states, trajectory.actions, strict=True)
    ]


def optimize_on_elite(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    elite: list[Trajectory],
    *,
    epochs: int,
    gradient_clip_norm: float,
    device: torch.device,
    rng: random.Random,
    batch_size: int = 256,
) -> float:
    pairs = _training_pairs(elite)
    if not pairs:
        raise RuntimeError("elite trajectories contained no actions")
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            states = [item[0] for item in chunk]
            actions = [item[1] for item in chunk]
            candidates = [legal_edges_from_state(state) for state in states]
            batch = collate_graph_states(states, candidates).to(device)
            targets = action_indices(candidates, actions).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("encountered a non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _validate(
    model: nn.Module,
    *,
    sizes: tuple[int, ...],
    episodes: int,
    seed: int,
    global_step: int,
    r: int,
    temperature: float,
    device: torch.device,
    sampling_protocol: str = "episode-v2",
) -> dict[int, float]:
    results: dict[int, float] = {}
    for n in sizes:
        validation_seeds = [
            _rollout_seed(seed + 10_000, global_step, n, episode) for episode in range(episodes)
        ]
        trajectories = run_episode_batch(
            model,
            n=n,
            seeds=validation_seeds,
            r=r,
            temperature=temperature,
            device=device,
            sampling_protocol=sampling_protocol,
        )
        results[n] = sum(item.optimality_ratio for item in trajectories) / len(trajectories)
    return results


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_best(
    path: Path,
    *,
    model_state: dict[str, torch.Tensor],
    model_config: ModelConfig,
    seed: int,
    stage_size: int,
    iteration: int,
    validation_score: float,
    run_name: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    atomic_torch_save(
        {
            "provenance": provenance,
            "format_version": 1,
            "model_state": model_state,
            "model_config": asdict(model_config),
            "model_family": model_config.family,
            "seed": seed,
            "stage_size": stage_size,
            "iteration": iteration,
            "validation_score": validation_score,
            "run_name": run_name,
        },
        path,
    )


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checked_device = torch.device(device)
    payload = torch.load(Path(path), map_location=checked_device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model_config = ModelConfig(**payload["model_config"])
    if payload.get("model_family", model_config.family) != model_config.family:
        raise ValueError("checkpoint model_family conflicts with model_config.family")
    model = build_model(model_config).to(checked_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "n": trajectory.n,
        "r": trajectory.r,
        "optimality_ratio": trajectory.optimality_ratio,
        "actions": [list(edge) for edge in trajectory.actions],
        "final_graph": graph_to_dict(trajectory.final_state),
    }


def _latest_checkpoint(
    *,
    model: nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    seed: int,
    next_stage_index: int,
    next_iteration: int,
    global_step: int,
    selection_rng: random.Random,
    current_stage_index: int,
    consecutive: int,
    best_stage_score: float,
    best_stage_state: dict[str, torch.Tensor] | None,
    best_stage_optimizer_state: dict[str, Any] | None,
    best_stage_iteration: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state": _cpu_state_dict(model),
        "model_config": asdict(model_config),
        "model_family": model_config.family,
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "seed": seed,
        "next_stage_index": next_stage_index,
        "next_iteration": next_iteration,
        "global_step": global_step,
        "selection_rng_state": selection_rng.getstate(),
        "current_stage_index": current_stage_index,
        "consecutive": consecutive,
        "best_stage_score": best_stage_score,
        "best_stage_state": best_stage_state,
        "best_stage_optimizer_state": best_stage_optimizer_state,
        "best_stage_iteration": best_stage_iteration,
    }


def _train(
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Train one curriculum seed and return its best-checkpoint path."""
    config = load_training_config(config_path, seed_override)
    seed = config.training.seed
    device = resolve_device(config.run.device)
    torch.set_num_threads(config.run.num_threads)
    set_global_seed(seed, deterministic=config.run.deterministic)
    run_dir = Path(config.run.output_dir) / f"{config.run.name}-seed-{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    trajectories_path = run_dir / "sample_trajectories.jsonl"
    contract = config.as_dict()
    contract["run"].pop("output_dir")
    contract["run"].pop("name")
    # JSON canonicalization makes tuple/list differences irrelevant on reload.
    contract = json.loads(json.dumps(contract))
    payload = None
    if resume is not None:
        payload = torch.load(Path(resume), map_location=device, weights_only=False)
        if payload.get("resume_contract_version") != 2:
            raise ValueError("legacy checkpoint lacks a validated resume contract; start a new run")
        if payload.get("training_contract") != contract:
            raise ValueError(
                "resume training contract differs; optimizer/curriculum overrides rejected"
            )
    elif (run_dir / "metadata.json").exists():
        raise ValueError("run already exists; choose a new name or explicitly resume")

    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    selection_rng = random.Random(seed + 17)
    start_stage_index = 0
    start_iteration = 1
    global_step = 0
    resumed_stage_index = -1
    resumed_consecutive = 0
    resumed_best_score = -math.inf
    resumed_best_state: dict[str, torch.Tensor] | None = None
    resumed_best_optimizer_state: dict[str, Any] | None = None
    resumed_best_iteration = 1

    if resume is not None:
        assert payload is not None
        if payload.get("format_version") != 1 or payload.get("seed") != seed:
            raise ValueError("resume checkpoint format or seed does not match the configuration")
        checkpoint_config = payload.get("model_config")
        if checkpoint_config is not None and ModelConfig(**checkpoint_config) != config.model:
            raise ValueError("resume checkpoint model configuration does not match")
        if payload.get("model_family", config.model.family) != config.model.family:
            raise ValueError("resume checkpoint model family does not match")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_stage_index = int(payload["next_stage_index"])
        start_iteration = int(payload["next_iteration"])
        global_step = int(payload["global_step"])
        selection_rng.setstate(payload["selection_rng_state"])
        resumed_stage_index = int(payload["current_stage_index"])
        resumed_consecutive = int(payload["consecutive"])
        resumed_best_score = float(payload["best_stage_score"])
        resumed_best_state = payload["best_stage_state"]
        resumed_best_optimizer_state = payload.get("best_stage_optimizer_state")
        resumed_best_iteration = int(payload.get("best_stage_iteration", start_iteration))

    # No files are changed until checkpoint and optimizer compatibility passed.
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        metrics_path.write_text("", encoding="utf-8")
        trajectories_path.write_text("", encoding="utf-8")
        _write_resolved_toml(run_dir / "config.toml", config)
        snapshot = save_source_snapshot(run_dir / "source.tar.gz")
        write_json(
            run_dir / "metadata.json",
            {
                **environment_metadata(device),
                "seed": seed,
                "started_at_unix": time.time(),
                "configuration": config.as_dict(),
                "source_snapshot": snapshot,
                "sampling_protocol": config.run.sampling_protocol,
                "validation_seed_rule": (
                    "fresh panel keyed by training seed, global step, size, episode"
                ),
                "checkpoint_selection": "maximum mean validation across introduced sizes",
            },
        )
    else:
        assert payload is not None
        restore_random_state(payload["random_state"])
        if not (run_dir / "metadata.json").exists():
            raise ValueError("resume requires the original run directory and metadata")
        append_jsonl(
            run_dir / "resume_events.jsonl",
            {
                "resumed_at_unix": time.time(),
                "checkpoint": str(resume),
                "checkpoint_sha256": sha256_file(resume),
                "global_step": global_step,
                "environment": environment_metadata(device),
            },
        )
        if start_stage_index >= len(config.training.curriculum_sizes):
            return run_dir / "best.pt"
        # A saved checkpoint defines the committed iteration boundary. Remove
        # a possible trailing metrics record written before an interrupted save.
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        metrics_path.write_text(
            "".join(
                json.dumps(x, sort_keys=True) + "\n"
                for x in records
                if x["global_step"] <= global_step
            )
        )
    metadata = json.loads((run_dir / "metadata.json").read_text())

    latest_elite: list[Trajectory] = []
    curriculum = config.training.curriculum_sizes
    for stage_index in range(start_stage_index, len(curriculum)):
        current_size = curriculum[stage_index]
        introduced = curriculum[: stage_index + 1]
        iteration_start = start_iteration if stage_index == start_stage_index else 1
        if resumed_stage_index == stage_index:
            consecutive = resumed_consecutive
            best_stage_score = resumed_best_score
            best_stage_state = resumed_best_state
            best_stage_optimizer_state = resumed_best_optimizer_state
            best_stage_iteration = resumed_best_iteration
        else:
            consecutive = 0
            best_stage_score = -math.inf
            best_stage_state = None
            best_stage_optimizer_state = None
            best_stage_iteration = iteration_start

        for iteration in range(iteration_start, config.training.max_iterations_per_stage + 1):
            global_step += 1
            started = time.perf_counter()
            trajectories = _collect_rollouts(
                model,
                sizes=introduced,
                count=config.training.rollouts_per_iteration,
                seed=seed,
                global_step=global_step,
                r=config.r,
                temperature=config.training.temperature,
                sampling_protocol=config.run.sampling_protocol,
                device=device,
            )
            elite = select_elite_trajectories(
                trajectories, config.training.elite_fraction, selection_rng
            )
            latest_elite = elite
            loss = optimize_on_elite(
                model,
                optimizer,
                elite,
                epochs=config.training.optimization_epochs,
                gradient_clip_norm=config.training.gradient_clip_norm,
                device=device,
                rng=selection_rng,
            )
            record: dict[str, Any] = {
                "event": "iteration",
                "global_step": global_step,
                "stage_index": stage_index,
                "stage_size": current_size,
                "iteration": iteration,
                "loss": loss,
                "rollout_mean_ratio": sum(t.optimality_ratio for t in trajectories)
                / len(trajectories),
                "elite_mean_ratio": sum(t.optimality_ratio for t in elite) / len(elite),
                "duration_seconds": time.perf_counter() - started,
                "rollout_episodes": len(trajectories),
                "sampled_actions": sum(len(t.actions) for t in trajectories),
                "optimizer_updates": math.ceil(sum(len(t.actions) for t in elite) / 256)
                * config.training.optimization_epochs,
                "actions_by_size": {
                    str(n): sum(len(t.actions) for t in trajectories if t.n == n)
                    for n in introduced
                },
            }

            should_validate = (
                iteration % config.training.validation_interval == 0
                or iteration == config.training.max_iterations_per_stage
            )
            stage_done = False
            if should_validate:
                validation = _validate(
                    model,
                    sizes=introduced,
                    episodes=config.training.validation_episodes,
                    seed=seed,
                    global_step=global_step,
                    r=config.r,
                    temperature=config.training.temperature,
                    sampling_protocol=config.run.sampling_protocol,
                    device=device,
                )
                validation_score = sum(validation.values()) / len(validation)
                record["validation"] = {str(key): value for key, value in validation.items()}
                record["validation_mean_ratio"] = validation_score
                if validation_score > best_stage_score:
                    best_stage_score = validation_score
                    best_stage_state = _cpu_state_dict(model)
                    best_stage_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_stage_iteration = iteration
                if validation[current_size] >= config.training.advancement_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                stage_done = consecutive >= config.training.advancement_patience

            budget_exhausted = iteration == config.training.max_iterations_per_stage
            stage_done = stage_done or budget_exhausted
            record["stage_gate_passed"] = consecutive >= config.training.advancement_patience
            record["stage_complete"] = stage_done
            record["iteration_wall_seconds"] = time.perf_counter() - started
            append_jsonl(metrics_path, record)

            # A predeclared equal-rollout-budget snapshot, before stage restoration.
            if global_step == 50:
                atomic_torch_save(
                    {
                        "format_version": 1,
                        "checkpoint_kind": "fixed-budget-current-model",
                        "global_step": global_step,
                        "model_state": _cpu_state_dict(model),
                        "model_config": asdict(config.model),
                        "model_family": config.model.family,
                        "seed": seed,
                        "provenance": {"configuration": config.as_dict(), "environment": metadata},
                    },
                    run_dir / "budget-0050.pt",
                )

            if stage_done:
                if best_stage_state is None:
                    raise RuntimeError("stage completed without a validation checkpoint")
                model.load_state_dict(best_stage_state)
                if best_stage_optimizer_state is not None:
                    optimizer.load_state_dict(best_stage_optimizer_state)

            next_stage = stage_index + 1 if stage_done else stage_index
            next_iteration = 1 if stage_done else iteration + 1
            atomic_torch_save(
                {
                    "resume_contract_version": 2,
                    "training_contract": contract,
                    "random_state": capture_random_state(),
                    **_latest_checkpoint(
                        model=model,
                        model_config=config.model,
                        optimizer=optimizer,
                        seed=seed,
                        next_stage_index=next_stage,
                        next_iteration=next_iteration,
                        global_step=global_step,
                        selection_rng=selection_rng,
                        current_stage_index=stage_index,
                        consecutive=consecutive,
                        best_stage_score=best_stage_score,
                        best_stage_state=best_stage_state,
                        best_stage_optimizer_state=best_stage_optimizer_state,
                        best_stage_iteration=best_stage_iteration,
                    ),
                },
                run_dir / "latest.pt",
            )
            if stage_done:
                stage_path = run_dir / f"stage-n{current_size}.pt"
                _save_best(
                    stage_path,
                    model_state=best_stage_state,
                    model_config=config.model,
                    seed=seed,
                    stage_size=current_size,
                    iteration=best_stage_iteration,
                    validation_score=best_stage_score,
                    run_name=config.run.name,
                    provenance={
                        "configuration": config.as_dict(),
                        "environment": metadata,
                        "selected_global_step": global_step - iteration + best_stage_iteration,
                        "sampling_protocol": config.run.sampling_protocol,
                    },
                )
                _save_best(
                    run_dir / "best.pt",
                    model_state=best_stage_state,
                    model_config=config.model,
                    seed=seed,
                    stage_size=current_size,
                    iteration=best_stage_iteration,
                    validation_score=best_stage_score,
                    run_name=config.run.name,
                    provenance={
                        "configuration": config.as_dict(),
                        "environment": metadata,
                        "selected_global_step": global_step - iteration + best_stage_iteration,
                        "sampling_protocol": config.run.sampling_protocol,
                    },
                )
                break

    sample_count = config.artifacts.trajectory_sample_count
    for trajectory in sorted(latest_elite, key=lambda item: item.optimality_ratio, reverse=True)[
        :sample_count
    ]:
        append_jsonl(trajectories_path, _trajectory_record(trajectory))
    write_json(
        run_dir / "completion.json",
        {
            "completed_at_unix": time.time(),
            "global_steps": global_step,
            "best_checkpoint": str(run_dir / "best.pt"),
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        },
    )
    return run_dir / "best.pt"


def train(
    config_path: str | Path, *, seed_override: int | None = None, resume: str | Path | None = None
) -> Path:
    """Run validated training and persist numerical failures without fabricating scores."""
    try:
        return _train(config_path, seed_override=seed_override, resume=resume)
    except FloatingPointError as error:
        config = load_training_config(config_path, seed_override)
        path = Path(config.run.output_dir) / f"{config.run.name}-seed-{config.training.seed}"
        write_json(
            path / "failure.json",
            {"status": "numerical_failure", "error": str(error), "time": time.time()},
        )
        raise

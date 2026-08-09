"""Typed TOML configuration loading."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    output_dir: str
    device: str = "cpu"
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    family: Literal["gnn", "mlp"] = "gnn"
    node_feature_dim: int = 4
    candidate_feature_dim: int = 5
    hidden_dim: int = 64
    message_passing_layers: int = 3
    max_nodes: int = 24


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    curriculum_sizes: tuple[int, ...]
    rollouts_per_iteration: int
    temperature: float
    elite_fraction: float
    learning_rate: float
    weight_decay: float
    optimization_epochs: int
    gradient_clip_norm: float
    validation_interval: int
    validation_episodes: int
    advancement_threshold: float
    advancement_patience: int
    max_iterations_per_stage: int
    seed: int


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    trajectory_sample_count: int = 10


@dataclass(frozen=True, slots=True)
class FullTrainingConfig:
    run: RunConfig
    r: int
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    name: str
    run_dir: str
    checkpoint_glob: str
    output_csv: str
    sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    episodes_per_method: int
    methods: tuple[str, ...]
    device: str
    model: ModelConfig
    mlp_checkpoint_glob: str | None = None
    mlp_model: ModelConfig | None = None


def _read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_training_config(path: str | Path, seed_override: int | None = None) -> FullTrainingConfig:
    value = _read_toml(path)
    try:
        run = RunConfig(**value["run"])
        model = ModelConfig(**value["model"])
        training_values = dict(value["training"])
        training_values["curriculum_sizes"] = tuple(training_values["curriculum_sizes"])
        if seed_override is not None:
            training_values["seed"] = seed_override
        training = TrainingConfig(**training_values)
        artifacts = ArtifactConfig(**value.get("artifacts", {}))
        r = int(value["environment"]["r"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid training configuration: {error}") from error
    _validate_training(run, r, model, training, artifacts)
    return FullTrainingConfig(run=run, r=r, model=model, training=training, artifacts=artifacts)


def _validate_training(
    run: RunConfig,
    r: int,
    model: ModelConfig,
    training: TrainingConfig,
    artifacts: ArtifactConfig,
) -> None:
    if not run.name or not run.output_dir:
        raise ValueError("run name and output_dir are required")
    if r != 2:
        raise ValueError("the MVP supports only environment.r = 2")
    if model.node_feature_dim != 4 or model.candidate_feature_dim != 5:
        raise ValueError("the MVP feature dimensions are fixed at 4 node and 5 candidate features")
    if model.family not in {"gnn", "mlp"}:
        raise ValueError("model.family must be 'gnn' or 'mlp'")
    if model.hidden_dim < 1 or model.message_passing_layers < 1:
        raise ValueError("model hidden width and depth must be positive")
    if model.max_nodes < 2:
        raise ValueError("model.max_nodes must be at least 2")
    if not training.curriculum_sizes or any(size < 2 for size in training.curriculum_sizes):
        raise ValueError("curriculum_sizes must contain values >= 2")
    if tuple(sorted(set(training.curriculum_sizes))) != training.curriculum_sizes:
        raise ValueError("curriculum_sizes must be strictly increasing")
    if model.family == "mlp" and max(training.curriculum_sizes) > model.max_nodes:
        raise ValueError("MLP curriculum sizes cannot exceed model.max_nodes")
    positive_values = (
        training.rollouts_per_iteration,
        training.optimization_epochs,
        training.validation_interval,
        training.validation_episodes,
        training.advancement_patience,
        training.max_iterations_per_stage,
    )
    if any(value < 1 for value in positive_values):
        raise ValueError("training counts and intervals must be positive")
    if training.rollouts_per_iteration < len(training.curriculum_sizes):
        raise ValueError("rollouts_per_iteration must be at least the number of curriculum sizes")
    if not 0 < training.elite_fraction <= 1:
        raise ValueError("elite_fraction must lie in (0, 1]")
    if training.temperature <= 0 or training.learning_rate <= 0:
        raise ValueError("temperature and learning_rate must be positive")
    if not 0 <= training.advancement_threshold <= 1:
        raise ValueError("advancement_threshold must lie in [0, 1]")
    if artifacts.trajectory_sample_count < 0:
        raise ValueError("trajectory_sample_count cannot be negative")


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    value = _read_toml(path)
    try:
        raw = dict(value["evaluation"])
        raw["sizes"] = tuple(raw["sizes"])
        raw["seeds"] = tuple(raw["seeds"])
        raw["methods"] = tuple(raw["methods"])
        mlp_model = ModelConfig(**value["mlp_model"]) if "mlp_model" in value else None
        config = EvaluationConfig(
            **raw,
            model=ModelConfig(**value["model"]),
            mlp_model=mlp_model,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid evaluation configuration: {error}") from error
    supported = {
        "gnn",
        "untrained_gnn",
        "mlp",
        "untrained_mlp",
        "random",
        "least_degree",
        "turan_oracle",
    }
    unknown = set(config.methods) - supported
    if unknown:
        raise ValueError(f"unknown evaluation methods: {sorted(unknown)}")
    if not config.sizes or any(size < 2 for size in config.sizes):
        raise ValueError("evaluation sizes must contain values >= 2")
    if not config.seeds or config.episodes_per_method < 1:
        raise ValueError("evaluation seeds and a positive episode count are required")
    if config.model.family != "gnn":
        raise ValueError("evaluation model.family must be 'gnn'")
    mlp_methods = {"mlp", "untrained_mlp"} & set(config.methods)
    if mlp_methods and config.mlp_model is None:
        raise ValueError("MLP evaluation methods require an [mlp_model] section")
    if "mlp" in config.methods and not config.mlp_checkpoint_glob:
        raise ValueError("the trained MLP method requires evaluation.mlp_checkpoint_glob")
    if config.mlp_model is not None:
        if config.mlp_model.family != "mlp":
            raise ValueError("mlp_model.family must be 'mlp'")
        if max(config.sizes) > config.mlp_model.max_nodes:
            raise ValueError("MLP evaluation sizes cannot exceed mlp_model.max_nodes")
    return config

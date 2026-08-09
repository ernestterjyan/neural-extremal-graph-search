# Configuration

Training and evaluation use TOML files. Relative output paths are resolved
from the repository working directory.

## Training sections

- `[run]`: run name, output directory, device, and deterministic mode.
- `[environment]`: fixed MVP value `r = 2`.
- `[model]`: `family = "gnn"` or `family = "mlp"`, feature dimensions, and
  hidden width. GNNs use `message_passing_layers`; MLPs use `max_nodes` as a
  hard padded capacity.
- `[training]`: curriculum, rollout counts, elite fraction, optimizer values,
  validation cadence, stage gate, stage budget, and seed.

Command-line `--seed` overrides `training.seed`. `device = "cpu"` is the
canonical setting. `device = "auto"` chooses MPS when available, then CPU.

## Evaluation sections

- `[evaluation]`: graph sizes, seeds, episodes per method, methods, checkpoint
  globs, output path, and device. `checkpoint_glob` selects GNN checkpoints;
  `mlp_checkpoint_glob` selects MLP checkpoints when `mlp` is requested.
- `[model]`: must match the trained checkpoint architecture.
- `[mlp_model]`: required by `mlp` or `untrained_mlp` and must include
  `family = "mlp"` plus the fixed capacity.

The supported method names are `gnn`, `untrained_gnn`, `mlp`,
`untrained_mlp`, `random`, `least_degree`, and `turan_oracle`.

## Fast development configurations

`smoke.toml` and `smoke-eval.toml` validate the GNN integration.
`mlp-smoke.toml` and `mlp-smoke-eval.toml` do the same for the fixed-size
control. These tiny budgets do not produce research-quality evidence.

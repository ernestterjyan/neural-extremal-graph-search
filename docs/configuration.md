# Configuration

Training and evaluation use TOML files. Relative output paths are resolved
from the repository working directory.

## Training sections

- `[run]`: run name, output directory, device, and deterministic mode.
- `[environment]`: fixed MVP value `r = 2`.
- `[model]`: input width, hidden width, message-passing depth, and candidate
  feature width.
- `[training]`: curriculum, rollout counts, elite fraction, optimizer values,
  validation cadence, stage gate, stage budget, and seed.

Command-line `--seed` overrides `training.seed`. `device = "cpu"` is the
canonical setting. `device = "auto"` chooses MPS when available, then CPU.

## Evaluation sections

- `[evaluation]`: graph sizes, seeds, episodes per method, methods, checkpoint
  glob, output path, and device.
- `[model]`: must match the trained checkpoint architecture.

The supported method names are `gnn`, `untrained_gnn`, `random`,
`least_degree`, and `turan_oracle`.

## Fast development configurations

`smoke.toml` and `smoke-eval.toml` use tiny budgets. They validate integration
but do not produce research-quality evidence.

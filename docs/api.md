# Public API

The stable import surface is re-exported from `extremal_graph`.

## Mathematical core

- `turan_partition_sizes(n, r)`
- `turan_edge_count(n, r)`
- `construct_turan(n, r)`

## Environment

- `GraphConstructionEnv`
- `GraphState`
- `Transition`
- `Edge`

The mutable environment uses optimized neighbour bitsets. `state` returns an
immutable snapshot suitable for serialization and model feature construction.

## Verification and serialization

- `verify_graph(graph, r, require_maximal=True)`
- `VerificationReport`
- `save_graph(graph, path)`
- `load_graph(path)`

Verification intentionally does not reuse the optimized legality routine.

## Neural policy and rollouts

- `GraphTensorBatch`
- `EdgePolicy`
- `FixedSizeMLPPolicy`
- `sample_actions`
- `Trajectory`
- `run_episode`
- `run_episode_batch`

`EdgePolicy` accepts padded dynamic batches; neither parameter shapes nor
features depend on a fixed graph size. `FixedSizeMLPPolicy` deliberately uses
absolute padded vertex positions and rejects inputs above its declared
`max_nodes` capacity.

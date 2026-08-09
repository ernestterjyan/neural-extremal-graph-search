# Fixed-size MLP baseline

Issue [#3](https://github.com/ernestterjyan/neural-extremal-graph-search/issues/3)
tracks the first v0.2 control experiment. This document fixes the model contract
before full training begins.

## Purpose

The control asks whether the v0.1 GNN's size transfer comes specifically from
permutation-equivariant message passing. The MLP reuses the same environment,
legal-action mask, node and candidate statistics, rollout sampler, curriculum,
optimizer, verifier, and metrics. Its intended difference is architectural.

## Fixed representation

The MLP declares `max_nodes = 24`. Every graph is padded to that size inside the
model, then represented by:

- the upper triangle of the fixed `24 x 24` adjacency matrix;
- all `24 x 4` padded node features;
- a 24-element node-presence mask.

A two-layer MLP maps this global fixed vector to one graph embedding. Each
candidate is scored from that embedding, the existing five symmetric candidate
statistics, and a 24-element sum of endpoint one-hot vectors.

The full control uses width 67: 37,990 trainable parameters versus 38,337 for
the release GNN, a difference below 1%. Parameter counts are also written into
new evaluation rows and aggregate tables rather than relying on this document.

The endpoint sum preserves the undirected edge convention, but its weights are
tied to absolute padded positions. Relabelling vertices therefore need not
preserve probabilities. Positions above the training range receive no direct
training signal, which is part of the fixed-size control rather than hidden by
shared equivariant parameters.

## Compatibility rules

- `family = "gnn"` remains the default, so every v0.1 TOML file is valid.
- Legacy v0.1 checkpoints without `family` or `max_nodes` load as GNNs through
  dataclass defaults.
- New checkpoints store the complete model configuration and an explicit
  `model_family` field.
- MLP configuration rejects curriculum or evaluation sizes above `max_nodes`.
- The existing `GraphTensorBatch` and candidate-mask contract are unchanged.

## Completed study

All five CPU seeds and 56,000 combined evaluation episodes completed. The MLP
reached 91.55% mean optimality and 17.3% exact success at `n = 24`, compared
with 91.82% and 29.5% for the GNN. It produced zero constraint violations and
all outputs were terminal-maximal. The full interpretation, including missed
curriculum gates, is in [`reports/v0.2_report.md`](../reports/v0.2_report.md).

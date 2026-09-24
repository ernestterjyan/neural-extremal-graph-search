# First neural solver pilot on C4-free graphs

The [separately frozen protocol](protocol.json) trains new GNN and endpoint policies on C4-free construction at 20 vertices, using the same rollouts, elite-prefix examples and optimizer updates. It evaluates their selected checkpoints through 40 vertices against the published exact values and the [nonlearned C4 pilot](../c4_pilot/README.md). The goal is to test whether this neural formulation is promising enough to develop further, not to claim a better solver in advance.

Run from the repository root with the locked Python environment:

```sh
python experiments/c4_neural_pilot.py train
python experiments/c4_neural_pilot.py evaluate
python experiments/c4_neural_pilot.py verify
python experiments/c4_neural_pilot.py report
```

Training logs and selected checkpoints live under `artifacts/`; each evaluated final graph and its seed is retained in `results.jsonl`. The outcome and limitations are recorded in the C4 pilot assessment after the panel is complete.

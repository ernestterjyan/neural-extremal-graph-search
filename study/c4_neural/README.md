# First neural solver pilot on C4-free graphs

The [separately frozen protocol](protocol.json) trains new GNN and endpoint policies on C4-free construction at 20 vertices, using the same rollouts, elite-prefix examples and optimizer updates. It evaluates their selected checkpoints through 40 vertices against the published exact values and the [nonlearned C4 pilot](../c4_pilot/README.md). The goal is to test whether this neural formulation is promising enough to develop further, not to claim a better solver in advance.

**Outcome:** The GNN did not beat the polarity construction at any evaluated size, and neither learned family reached an exact optimum in the 1,500 graphs it generated. Read the [full research note](RESEARCH_NOTE.md) for the results and limits.

Run from the repository root with the locked Python environment:

```sh
python experiments/c4_neural_pilot.py train
python experiments/c4_neural_pilot.py evaluate
python experiments/c4_neural_pilot.py verify
python experiments/c4_neural_pilot.py report
python experiments/analyze_c4_neural.py
python experiments/audit_c4_neural.py
python experiments/package_c4_neural.py package
python experiments/package_c4_neural.py check
python experiments/check_c4_checkout.py
```

Training logs and selected checkpoints live under `artifacts/` during execution and in the checksummed `training-evidence-v1.tar.gz` archive for portability. Each evaluated final graph and its seed is retained in `results.jsonl`. The clean-checkout check verifies the archive, revalidates the graphs, regenerates reports and audits, and runs the test suite using the installed Python environment.

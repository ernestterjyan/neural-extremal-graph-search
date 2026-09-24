# Parity mechanism follow-up

This is a separate exploratory study following the corrected transfer experiment. It holds all previously trained checkpoints fixed and gives five policies the same explicit bipartite-preservation mask. The primary question is whether the frozen GNN achieves higher exact success at n=40 than the frozen endpoint scorer when both receive that mask. The endpoint model has no message passing. The candidate-only scorer, uniform policy, and an explicit myopic balancing rule provide further controls.

`protocol.json` fixes the methods, 5 original training seeds, 4 sizes, 100 fresh evaluation episodes per cell, one primary contrast, and the seed rule before any comparison output is generated. The balancing control uses component color-class sizes after a proposed edge. It is an information-using heuristic, not a learned policy or exact balanced-completion oracle. Existing checkpoints were trained with unequal realized budgets; this frozen-weight comparison alone cannot causally isolate message passing.

From the repository root with the locked Python environment:

```sh
python experiments/parity_followup.py pilot
python experiments/parity_followup.py run
python experiments/parity_followup.py verify
python experiments/analyze_parity_followup.py
```

The pilot uses excluded seeds and records only timing. Evaluation cells are atomic and resumable under hashes of the protocol, generator, core source, and checkpoint. `verify` requires every planned cell, independently validates all persisted final graphs, and writes `artifacts/evaluation.csv`. Paired seed-level comparisons and the final interpretation are generated only after verification. Large graph chunks live under `artifacts/` and are distributed separately from ordinary source files.

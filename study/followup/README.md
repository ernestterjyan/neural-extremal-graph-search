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

## Matched exposure and updates

`matched_protocol.json` defines a second, explanatory training study. It trains all four families from scratch at n=14 for exactly 100 iterations and 25,600 terminal rollouts per seed. Every iteration selects 52 elite trajectories, takes their first 12 action examples, and performs exactly eight updates with fixed batch size 256. Validation is 100 fresh n=14 episodes every five iterations. The best validation checkpoint is evaluated on new episode seeds at n=14, 24 and 40; the MLP remains unsupported at n=40. Five original training-seed identifiers are reused only as replication labels, not weights or random episode streams.

```sh
python experiments/matched_exposure.py train
python experiments/matched_exposure.py evaluate
python experiments/matched_exposure.py verify
python experiments/analyze_matched_exposure.py
```

This new fixed-size training variant matches rollout, size, validation, selected action-pair and optimizer-update counts. It does not match the total number of environment actions, which depends on trajectory length, or the information and capacity available to each architecture. The n=24/40 benchmarks have already been inspected in the earlier study. Its purpose is to test whether the previous training-budget imbalance alone plausibly explains the architecture comparison.

## Look-ahead reachability

`lookahead_protocol.json` specifies an exhaustive search over all labeled states reachable through every action tied for the one-step heuristic's best score, up to eight vertices. A reachable destructive action would be an exact counterexample for that tie choice. Absence through the finite panel is not a general guarantee. Run `python experiments/lookahead_reachability.py` to regenerate the saved search record.

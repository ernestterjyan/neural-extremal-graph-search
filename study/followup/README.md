# Parity mechanism follow-up

This is a separate exploratory study following the corrected transfer experiment. It holds all previously trained checkpoints fixed and gives five policies the same explicit bipartite-preservation mask. The primary question is whether the frozen GNN achieves higher exact success at n=40 than the frozen endpoint scorer when both receive that mask. The endpoint model has no message passing. The candidate-only scorer, uniform policy, and an explicit myopic balancing rule provide further controls.

`protocol.json` fixes the methods, 5 original training seeds, 4 sizes, 100 fresh evaluation episodes per cell, one primary contrast, and the seed rule before any comparison output is generated. The balancing control uses component color-class sizes after a proposed edge. It is an information-using heuristic, not a learned policy or exact balanced-completion oracle. Existing checkpoints were trained with unequal realized budgets; this frozen-weight comparison alone cannot causally isolate message passing.

The balancing rule attained the optimum on all 2,000 declared episodes. [BALANCING_PROOF.md](BALANCING_PROOF.md) gives an elementary argument that, with this parity mask, it reaches the optimum for every n≥2 under any tie choice. The stronger baseline sharpens the scientific question: what learned behavior remains after accounting for explicit component balance information?

From the repository root with the locked Python environment:

```sh
python experiments/parity_followup.py pilot
python experiments/parity_followup.py run
python experiments/parity_followup.py verify
python experiments/analyze_parity_followup.py
```

The pilot uses excluded seeds and records only timing. Evaluation cells are atomic and resumable under hashes of the protocol, generator, core source, and checkpoint. `verify` requires every planned cell, independently validates all persisted final graphs, and writes `artifacts/evaluation.csv`. Paired seed-level comparisons and the final interpretation are generated only after verification. Large graph chunks live under `artifacts/` during generation and are also packaged in the completed checksummed `evidence-v1.tar.gz` archive.

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

[LOOKAHEAD_LIMIT.md](LOOKAHEAD_LIMIT.md) records the finite exhaustive result and a different, independently checked six-vertex partial-state counterexample. The latter does not show failure from the empty graph.

## Final evidence checks

After both panels finish, regenerate the reports and independently replay saved trajectories:

```sh
python experiments/audit_followup.py
python experiments/audit_matched_followup.py
python experiments/package_followup.py package
python experiments/package_followup.py check
python experiments/check_followup_checkout.py
```

The [research note](RESEARCH_NOTE.md) states the conclusions and claim boundaries. The frozen-weight parity comparison also requires the corrected study's `study/evidence-corrected-v1.tar.gz` archive; the clean-checkout check extracts and verifies both archives. The bundle manifests and clean-checkout validation record describe exactly which artifacts were checked. The clean checkout reuses the installed Python environment and does not rerun all twenty training jobs.

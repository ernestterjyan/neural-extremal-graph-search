# Execution record

- Starting repository: clean v0.2 base `87a6bc5a52f973b8025bdd5efeb0791fcc4dcb01`; original local checkout and audit evidence accessible.
- Isolated branch: `research/corrected-transfer-study`; repairs and frozen protocol committed before canonical training as `4b14e4b`.
- Read-only audit inputs: project-state, implementation, evidence and research audits, probes, verification records, historical replay and diagnostic outputs.
- Repairs: explicit legal masks/numerical failure handling; independent episode streams with named historical protocol; strict resolved resume contracts and preserved metadata; atomic checkpoints/RNG state; source/environment recording.
- Controls: candidate statistics only, endpoint encoder without message passing, independently tested look-ahead.
- Profiling: n40 eight-episode GNN batch improved from 5.91 to 1.93 seconds after batching feature tensor creation; replay regressions remain unchanged. Timing-only pilot, excluded seeds, no comparative-quality tuning.
- Protocol: five seeds, four learned families, original 100-iteration stage caps, eighteen evaluation sizes, 100 episodes per cell; four one-thread CPU workers; fixed 50-iteration comparison.
- Validation before launch: 60 tests passed; lint and format passed.
- Bookkeeping deviation: runner initially attempted `relative_to` on a relative checkpoint path after successful training completion. Checkpoints/completion records remained intact. The runner was corrected to resolve paths; the ledger preserves original errors and reconciliation. No training/source-module change or seed replacement.
- Historical replication: 640 saved n24 outputs independently verified; every edge count matched the published CSV.
- Constructed-state study: six exact continuation collections enumerated; C8 values and uniform-action limitation reproduced at depths 0, 1, 3 and 8 across five initializations.

Final completion counts, resource usage and evidence conclusions are recorded in `RESEARCH_REPORT.md`, `verification.json`, `report_inputs.json`, and the durable ledger. This record distinguishes engineering/replay checks from new research observations.

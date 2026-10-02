# Independent evidence audit and input replay

`experiments/c4_repair_audit.py` uses only the Python standard library. It does not
import the solver or its graph implementation. It reconstructs neighbor sets and
checks every retained initial, repaired, starting-pool, incumbent-event and timed
history witness for simplicity and the common-neighbor C4 condition.

It also checks source snapshots, protocol hashes, declared experimental cells and
seed schedules, fixed complements, gain consistency, strictly improving histories,
checkpoint availability, matched starting pools and model/block identity. A completed
search must contain its graph, history, checkpoints, configuration and timing fields.
Primary/validation panels must include GNN, MLP, adaptive and tabu, eight blocks and
ten searches per size/block. Partial checks cannot authorize a gate or inference.

The audit checks stored timestamp consistency, not an independently remeasured
execution clock. It checks local-optimum claims against solver status, not an external
CP-SAT proof certificate. Read `FORMULATION_AUDIT.md` for the initial-constraint
comment error and the disclosed equal-score replacement-witness limitation.

## Completed first feasibility panel

The retained independent audit versions all confirm the original outcome:
**5/100 improved states, no failed states, gate failed**. The current retained
`feasibility_v1/INDEPENDENT_AUDIT_v3.json` checks 2,297 graph occurrences and 527
local-certificate/status claims. Its 1,797 repair records predate event-level timing;
the audit explicitly marks that timing evidence as legacy. Earlier audit programs
and reports are retained with their exact source hashes rather than overwritten.

`feasibility_v1/INPUT_REPLAY.json` additionally records successful replay of **all
100 starting states and complete ordered candidate pools**, using the archived source
and original seed schedules. This catches omitted candidate labels as well as changed
inputs. It does not rerun repair optimization or independently reproduce wall-clock
performance. The separate six-cell repair replay in `reproduction_v1` does rerun its
declared original-source subset in a fresh locked virtual environment on the same host.

Each audit report retains its auditor program. Each input-replay report retains both
its replay program and auditor dependency. A retained replay program loads its own
auditor and the batch's archived solver source, without using the live solver checkout.

## Commands

Choose a new output filename for each audit; existing evidence is never replaced.

```sh
.venv/bin/python experiments/c4_repair_audit.py --batch study/c4_repair/BATCH --output study/c4_repair/BATCH/INDEPENDENT_AUDIT.json
.venv/bin/python experiments/c4_repair_replay_inputs.py --batch study/c4_repair/BATCH --output study/c4_repair/BATCH/INPUT_REPLAY.json
```

Input replay covers feasibility, label collection and development profiles. It does
not claim deterministic reproduction of timed search trajectories. `--allow-partial`
checks only completed lines. Replay uses exactly the audited byte prefix, so a live
append or unfinished line cannot silently enter the checked result or imply completion.

The new integrity tests reject missing final graphs, late gains, changed fixed
complements, wrong provenance, duplicate cells, C4 witnesses, altered source and
omitted candidates. They also exercise partial input and the retained standalone
replay program against real archived development evidence.

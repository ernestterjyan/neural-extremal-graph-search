# Local cost projection before paid provisioning

No paid resources have been created. Paid expenditure is €0. The monetary ceiling
is €500; development effort and local research time are recorded separately.

| Workload | Declared search time, single CPU worker |
|---|---:|
| One feasibility panel, 100 states × at most 18 regions × 1 s | at most 0.50 h, plus generation/verification |
| Revised feasibility panel, 100 states × at most 36 regions × 2 s | at most 2.00 h, plus generation/verification |
| Example labels, (6+5) sizes × 16 states × 18 regions × 1 s | at most 0.88 h, plus generation/verification |
| Labels using revised recipe, (6+5) sizes × 16 states × 36 regions × 2 s | at most 3.52 h, plus generation/verification |
| Completed classical tuning, 4 sizes × 3 seeds × 8 recipes × 60 s | 1.60 h, measured |
| Development calibration, 4 sizes × 3 seeds × 3 methods × 60 s | 0.60 h |
| End-to-end validation, 5 sizes × 8 blocks × 10 starts × 4 methods × 60 s | 26.67 h |
| Controlled evaluation, 8 sizes × 8 blocks × 10 starts × 4 methods × 60 s | 42.67 h |
| Each declared timed reproduction subset, 3 sizes × 4 methods × 60 s | 0.20 h |

These are worker-time budgets, not measured completion times or cloud price quotes.
Training, tuning, retries, record attempts, evidence storage, reproduction, and
deadline overshoot are additional. Training cost must be profiled on actual data.
Many small repairs finish before their limit, so label collection may cost less.
Gate failure stops downstream allocations.

The example breadth may be reduced before freezing. The final eight trained-model
seeds and ten starting/search seeds per block and size must remain intact.

Before any paid resources: choose a concrete provider/CPU/storage configuration,
verify its current prices, project the full workload including overhead, state the
maximum charge, obtain approval, and enter actual charges in `costs.jsonl`.

The fixed-budget edge-count experiment does not itself measure time-to-target.
Training/tuning break-even requires an additional declared development target and
measured time saved per search. If there is no demonstrated time saving, report
break-even as unsupported or infinite; do not infer it from faster model inference
alone. The controlled 1/10/60 s checkpoint data can support bounds on time-to-target
only where they actually bracket a shared target.

Before neural validation, the implementation prospectively declares a separate
**inference-only scenario**: cached selection on identical region pools at
n20/35/40, all sixteen trained checkpoints and the initial adaptive selector,
twenty timed repetitions after one warmup, seeded condition order. Include feature
extraction, ranking and choosing a candidate. Exclude common candidate generation
and checkpoint loading here; both remain charged in controlled search.

Average the three graph-level median latencies per model and compare with adaptive.
Assume thirty selection decisions per hypothetical search, explicitly as a
scenario, not an observed trajectory count. If the saving per decision is positive,
divide model-training seconds (and, separately, training plus all shared label /
classical-tuning seconds) by thirty times that saving, rounding up. If the saving
is zero or negative, report no finite inference-only recovery count. Report each
model's scenario separately; do not add shared costs repeatedly to campaign totals.

These conditional counts are development timing diagnostics. They cannot establish
end-to-end acceleration, held-out transfer or the primary graph-quality result.
The actual ledger reports retained experiment/model time; human effort and all
orchestration overhead are outside that measured quantity. Final training/tuning
costs remain separately visible even when no saving is demonstrated.

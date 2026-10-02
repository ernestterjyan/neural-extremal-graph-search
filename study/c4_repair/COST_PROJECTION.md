# Local cost projection before paid provisioning

No paid resources have been created. Paid expenditure is €0. The monetary ceiling
is €500; development effort and local research time are recorded separately.

| Workload | Declared search time, single CPU worker |
|---|---:|
| One feasibility panel, 100 states × at most 18 regions × 1 s | at most 0.50 h, plus generation/verification |
| Revised feasibility panel, 100 states × at most 36 regions × 2 s | at most 2.00 h, plus generation/verification |
| Example labels, (6+5) sizes × 16 states × 18 regions × 1 s | at most 0.88 h, plus generation/verification |
| Labels using revised recipe, (6+5) sizes × 16 states × 36 regions × 2 s | at most 3.52 h, plus generation/verification |
| End-to-end validation, 5 sizes × 8 blocks × 10 starts × 4 methods × 60 s | 26.67 h |
| Controlled evaluation, 8 sizes × 8 blocks × 10 starts × 4 methods × 60 s | 42.67 h |

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

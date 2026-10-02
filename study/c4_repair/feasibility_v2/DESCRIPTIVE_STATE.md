# Independent descriptive state report

Stage: `feasibility`. Recorded cells: 100/100. Panel complete: **True**.

Failed cells retained: **0**.

This is a descriptive account. It does not establish a learned advantage, transfer,
a new extremal bound or global optimality. Missing/failed searches are visible below.

Feasibility: **PASS**; 28/100 states improved; threshold 10/100.

| Size | Completed | Improved | Best initial | Best repaired | Above best initial |
|---|---:|---:|---:|---:|---:|
| 35 | 50 | 16 | 102 | 102 | 0 |
| 40 | 50 | 12 | 125 | 125 | 0 |

Improving a weaker start is distinct from exceeding the strongest observed
construction. Neither comparison proves a mathematical optimum.

## Provenance and limits

Results prefix SHA256: `c03a113333604a884afef6015ec197c916a41e6a6233794f05196402eee02273`.

Source SHA256: `b723c0319a458bf833d8ad23fdd64154a3371c842591fb8f612cf790517f5c55`.

Protocol SHA256: `ec20414c295cbcd029ee1afc6d1359ba94b5ba68d8c4a6bb6022f0f868568cb3`.

- Set-based witness checks, not a separate CP optimality proof checker.
- Stored monotonic timestamps are checked for consistency, not externally remeasured.
- Repair equal-score replacement witnesses may lack an event; only the recorded improvement witness receives its timestamp.
- Descriptive report; controlled claims require the frozen paired block analysis.
- Incomplete panels and failed searches cannot be silently excluded from inference.
- Witness checks do not independently prove CP-SAT optimum certificates.

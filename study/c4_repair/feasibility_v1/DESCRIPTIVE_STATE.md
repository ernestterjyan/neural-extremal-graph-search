# Independent descriptive state report

Stage: `feasibility`. Recorded cells: 100/100. Panel complete: **True**.

Failed cells retained: **0**.

This is a descriptive account. It does not establish a learned advantage, transfer,
a new extremal bound or global optimality. Missing/failed searches are visible below.

Feasibility: **FAIL**; 5/100 states improved; threshold 10/100.

| Size | Completed | Improved | Best initial | Best repaired | Above best initial |
|---|---:|---:|---:|---:|---:|
| 35 | 50 | 1 | 102 | 102 | 0 |
| 40 | 50 | 4 | 125 | 125 | 0 |

Improving a weaker start is distinct from exceeding the strongest observed
construction. Neither comparison proves a mathematical optimum.

## Provenance and limits

Results prefix SHA256: `760c2aab96efb23d2f0eac608fb6ee13e6d096295d18624b5b32e034f60992f8`.

Source SHA256: `0056e529e14cd8668c9e0c6333c669f090135cee0600a9baefd66d577e350430`.

Protocol SHA256: `cc5383d918ab9f575003912d363e9618b428c1506d4f7554c4e44c49bf6e698f`.

- Set-based witness checks, not a separate CP optimality proof checker.
- Stored monotonic timestamps are checked for consistency, not externally remeasured.
- Repair equal-score replacement witnesses may lack an event; only the recorded improvement witness receives its timestamp.
- Descriptive report; controlled claims require the frozen paired block analysis.
- Incomplete panels and failed searches cannot be silently excluded from inference.
- Witness checks do not independently prove CP-SAT optimum certificates.

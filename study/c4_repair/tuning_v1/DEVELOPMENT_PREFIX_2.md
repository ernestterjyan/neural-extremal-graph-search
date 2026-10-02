# Independent descriptive state report

Stage: `development_tuning`. Recorded cells: 68/96. Panel complete: **False**.

Failed cells retained: **0**.

This is a descriptive account. It does not establish a learned advantage, transfer,
a new extremal bound or global optimality. Missing/failed searches are visible below.

## tabu/fixed_h5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 9 | 85.44444444444444 |
| 10 | 9 | 86.33333333333333 |
| 60 | 9 | 86.33333333333333 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 9 | 85.66666666666667 |
| 10 | 9 | 86.33333333333333 |
| 60 | 9 | 86.33333333333333 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h7_15

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 9 | 84.77777777777777 |
| 10 | 9 | 86 |
| 60 | 9 | 86.33333333333333 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h12_25

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 8 | 83 |
| 10 | 8 | 83.25 |
| 60 | 8 | 83.625 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 8 | 82.625 |
| 10 | 8 | 82.625 |
| 60 | 8 | 82.625 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation10

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 8 | 82.625 |
| 10 | 8 | 82.75 |
| 60 | 8 | 82.75 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/explore_stagnation10

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 9 | 84.66666666666667 |
| 10 | 9 | 84.77777777777777 |
| 60 | 9 | 84.77777777777777 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation20

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 8 | 82.625 |
| 10 | 8 | 82.75 |
| 60 | 8 | 82.75 |

These completed-cell means can be unbalanced in an incomplete panel.

## Provenance and limits

Results prefix SHA256: `f40cff22905b3358b220b3b55c14a0ecf136bb046ba4dd9b3e744c6a74dd7b43`.

Source SHA256: `7f9b5d259e057f6d1ba8d029e2fd4d91eb4426413bfc7433a3c024af96dde312`.

Protocol SHA256: `5651469eaebfe3269ef95ea06a913ae93be8810737bf23662c54d7b234fd26c0`.

- Set-based witness checks, not a separate CP optimality proof checker.
- Stored monotonic timestamps are checked for consistency, not externally remeasured.
- Repair equal-score replacement witnesses may lack an event; only the recorded improvement witness receives its timestamp.
- Descriptive report; controlled claims require the frozen paired block analysis.
- Incomplete panels and failed searches cannot be silently excluded from inference.
- Witness checks do not independently prove CP-SAT optimum certificates.

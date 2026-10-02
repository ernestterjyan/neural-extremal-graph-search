# Independent descriptive state report

Stage: `development_tuning`. Recorded cells: 16/96. Panel complete: **False**.

Failed cells retained: **0**.

This is a descriptive account. It does not establish a learned advantage, transfer,
a new extremal bound or global optimality. Missing/failed searches are visible below.

## tabu/fixed_h5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 63 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 63 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h7_15

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 63 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## tabu/reactive_h12_25

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 63 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation5

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 62.5 |
| 10 | 2 | 62.5 |
| 60 | 2 | 62.5 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation10

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 62.5 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/explore_stagnation10

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 62.5 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## adaptive/stagnation20

| Seconds | Completed searches | Mean edges |
|---:|---:|---:|
| 1 | 2 | 62.5 |
| 10 | 2 | 63 |
| 60 | 2 | 63 |

These completed-cell means can be unbalanced in an incomplete panel.

## Provenance and limits

Results prefix SHA256: `1c88047c422b252c85a162a0ecc756c27490edc1ccb58e9692505f3cce4aae05`.

Source SHA256: `7f9b5d259e057f6d1ba8d029e2fd4d91eb4426413bfc7433a3c024af96dde312`.

Protocol SHA256: `5651469eaebfe3269ef95ea06a913ae93be8810737bf23662c54d7b234fd26c0`.

- Set-based witness checks, not a separate CP optimality proof checker.
- Stored monotonic timestamps are checked for consistency, not externally remeasured.
- Repair equal-score replacement witnesses may lack an event; only the recorded improvement witness receives its timestamp.
- Descriptive report; controlled claims require the frozen paired block analysis.
- Incomplete panels and failed searches cannot be silently excluded from inference.
- Witness checks do not independently prove CP-SAT optimum certificates.

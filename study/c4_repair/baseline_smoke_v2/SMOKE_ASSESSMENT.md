# Two-second development smoke assessment

One paired starting/search seed at each size; matched initial pools and seeded method order.
All 12 searches completed without failures. The independent auditor checked 93 graph
occurrences and time-budget consistency. This is a smoke check, not a strength benchmark.

| n | Tabu | Adaptive | Construction | Simple repair |
|---|---:|---:|---:|---:|
| 20 | 46 | 46 | 46 | 46 |
| 35 | 106 | 102 | 102 | 102 |
| 40 | 124 | 124 | 125 | 124 |

The tabu n35 witness reached 106 edges at 1.173532s.
It additionally passed independent explicit four-cycle enumeration. This matches
the known exact n35 value in the [Afzaly–McKay table](https://users.cecs.anu.edu.au/~bdm/data/extremal.html),
previously checked 2026-10-02; it is not a new bound. This single draw is not evidence
of a general performance difference or learned superiority.

Construction can improve its initial pool by generating further seeds during the budget.
The n40 construction result of 125 therefore need not equal the 124-edge initial best
received by all four methods.

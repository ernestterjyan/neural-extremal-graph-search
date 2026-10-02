# Paired classical development tuning

Sixty seconds per search, matched starts; all trials retained.

| Trial | Mean at 1s | Mean at 10s | Mean at 60s |
|---|---:|---:|---:|
| tabu/fixed_h5 | 95.083 | 95.750 | 96.000 |
| tabu/reactive_h5 | 95.250 | 96.000 | 96.000 |
| tabu/reactive_h7_15 | 94.583 | 95.500 | 95.833 |
| tabu/reactive_h12_25 | 94.917 | 95.250 | 95.500 |
| adaptive/stagnation5 | 94.500 | 94.500 | 94.500 |
| adaptive/stagnation10 | 94.500 | 94.583 | 94.583 |
| adaptive/explore_stagnation10 | 94.500 | 94.583 | 94.583 |
| adaptive/stagnation20 | 94.500 | 94.583 | 94.583 |

Selected: {'tabu': 'tabu/reactive_h5', 'adaptive': 'adaptive/explore_stagnation10'}.

development tuning only; not evidence of held-out solver strength

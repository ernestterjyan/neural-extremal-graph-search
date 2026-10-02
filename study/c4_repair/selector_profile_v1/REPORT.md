# Untrained selector timing profile

Verified Apple M5; one Torch thread; full feature extraction and batched scoring.
Random untrained models: these timings establish no learned search benefit.

| n | Candidates | Candidate generation (ms) | GNN median (ms) | MLP median (ms) |
|---|---:|---:|---:|---:|
| 20 | 36 | 0.115 | 1.194 | 0.752 |
| 35 | 36 | 0.141 | 1.447 | 0.906 |
| 40 | 36 | 0.151 | 1.541 | 0.923 |

Twenty repetitions after one warmup per family and graph. Cold checkpoint loading is
not measured here; actual validation charges it and all inference to the search clock.
All three input graph witnesses passed the separate standard-library audit.

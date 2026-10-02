# Development tuning freeze

Frozen before observing any tuning outcome. Source and recipe are retained in
contract.json and source/. Commit ab684ea contains the tested integrated code.

96 searches: n25/31/35/40, three paired seeds, four tabu and four adaptive recipes,
60 seconds per search. Trials use shuffled order and identical initial pools.
Select by mean at 60s, then 10s, then 1s, then trial ID; retain every outcome.
The region recipe remains sizes 3/4/5, four draws per family/size, 2s per repair.

Verified Apple M5 CPU. One process, one CP-SAT/Torch worker. Nominal local cost:
1.60 worker-hours plus overhead; paid expenditure €0. This is development tuning
and cannot demonstrate held-out transfer or replace the learned primary result.

The original feasibility evidence and all evaluation sizes remain untouched.

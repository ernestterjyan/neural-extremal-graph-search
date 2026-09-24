# A harder candidate: maximum-edge C4-free graphs

This is a feasibility pilot for replacing the known-optimum triangle-free benchmark with the maximum-edge four-cycle-free problem. C4-free membership is cheap to verify, exact values are independently tabulated at the five chosen sizes, and the general extremal number is not known in closed form. The pilot is not a neural result or an equal-compute contest. It first asks whether simple and algebraic baselines leave useful headroom.

The [frozen protocol](protocol.json) declares five sizes, six methods and ten independent replications per method and size before results are generated. The published exact values used only for scoring come from [Afzaly and McKay's primary data](https://users.cecs.anu.edu.au/~bdm/data/extremal.html). Orthogonal-polarity graphs are established C4-free constructions; for general context see [He, Ma and Yang](https://arxiv.org/abs/1912.00986). A pilot win over the simple methods would not establish superiority to all known constructions or search programs.

From the repository root:

```sh
python experiments/c4_solver_pilot.py run
python experiments/c4_solver_pilot.py verify
python experiments/c4_solver_pilot.py report
```

All final graphs, seeds, construction times and code/protocol checksums are retained in `results.jsonl`; `summary.csv` is regenerated after verification. This branch keeps the C4 pilot separate from the corrected triangle-free study and its frozen evidence.

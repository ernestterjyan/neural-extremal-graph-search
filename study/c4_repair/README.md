# Learned repair campaign

This is new work alongside the frozen triangle-free studies and C4 construction
pilot. The research question is whether learning **where to repair** gives a
transferable advantage over strong classical search. No learned repair advantage
or new extremal bound has been established.

## Contract

- Development only: all orders up to 40; training 20, 24, 28, 32, 36, 40;
  validation 22, 26, 30, 34, 38.
- Untouched evaluation: 44, 48, 52, 56, 64, 72, 80, 96. Solver experiments on these
  orders require a frozen protocol, configuration, seeds, analysis and source.
- Eight independent training seeds per learned family, ten independent starting /
  search seeds per block and evaluation order. Do not treat graphs as model replicates.
- Primary: mean advantage >= 1 edge at 60 seconds, simultaneous 95% paired confidence
  intervals above zero against BOTH tabu and adaptive neighborhood search, and positive
  mean advantage on at least six of the eight orders. Report 1 and 10 seconds secondarily.
- Charge initialization, candidate generation, inference and repair to search time.
  Single CPU worker, controlled concurrency, identical starting pools. Record training /
  tuning costs and break-even searches separately.
- No published target-size witnesses in training or controlled starting pools.
- Feasibility gate: 100 development states below known optima at orders 35 and 40;
  at least 10% show independently verified improvement within the declared region pool
  and limits. Revise neighborhoods before training if it fails.
- Learning gate: require end-to-end validation benefit over adaptive selection after
  inference overhead. If it fails, stop the larger neural campaign and report honestly.
- Record attempts only after the primary comparison passes. Recheck literature,
  verify witnesses with common-neighbor counts and explicit four-cycle enumeration.

## Implementation and evidence

New code lives in `src/extremal_graph/repair/` and `experiments/c4_repair.py`.
The fixed complement of a selected vertex region stays unchanged; all incident
edges are repair variables. CP-SAT keeps a feasible incumbent and separates
remaining four-cycle constraints. A feasible timeout is **not** a local optimum;
a local optimum for a proper region is **not** a global extremal result.

Install with `uv sync --extra dev --extra repair`. Exact package versions are in
`uv.lock`. Timed searches cannot promise bit-identical trajectories on different
machines; deterministic work-limited tests and stored witness replay test the
parts that can be reproduced exactly.

See `PROGRESS.md` for the current milestone and `costs.jsonl` for spending.

## Reference inventory (checked 2026-10-02)

- [Afzaly–McKay data](https://users.cecs.anu.edu.au/~bdm/data/extremal.html):
  exact C4-only development values n=35:106 and n=40:127. Graph counts and edge
  counts are distinct columns. This table alone cannot certify a current record.
- [CP-SAT documentation](https://developers.google.com/optimization/cp/cp_solver):
  OPTIMAL, FEASIBLE and UNKNOWN have different meanings, particularly on timeout.
- [Mehrabian et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/772):
  neural/tabu graph search for **girth at least five** excludes both C3 and C4.
  Our C4-only objective permits triangles; its results are related work, not
  directly interchangeable benchmarks.

The original eight-week allocation is €100 infrastructure/feasibility, €200 data /
training, €120 controlled evaluation, €40 exploratory records, €40 reproduction /
writing. Total paid compute/storage cap €500. Local time is recorded separately.
Concrete paid provisioning requires approval; no paid resources have been created.

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

The revised feasibility panel passed **28/100**, with complete independent witness
audit and deterministic input replay. Best repaired graphs n35:102 and n40:125
matched the best initial constructions. This gate allows label collection with
the unchanged recipe (region sizes 3/4/5, four draws per family/size, two seconds per
repair); it does not demonstrate a learned advantage or a new lower bound.

The current runner includes marked-region GNN inference, family/size adaptive
selection, shared timed calibration, balanced method order, and a host-wide
process lease. See `DEVELOPMENT_CONTROLS.md` for the classical tuning contract.
There are no trained research selectors or controlled validation results yet.

## Next stages

Run commands from the repository root using its locked environment. All timed jobs
must run sequentially; the runner rejects a second campaign worker. Do not run tests
or other CPU-heavy work concurrently with timed experiments. Retain the exact source
snapshot for each batch and use that snapshot to resume long runs.

1. Profile selector inference using `experiments/c4_repair_selector_profile.py`.
   Its random untrained models provide timing evidence only. Run a small baseline
   workload with both classical and calibration methods; independently audit it.
2. Prepare `tuning_v1` with `prepare-tuning --feasibility
   study/c4_repair/feasibility_v2 --output study/c4_repair/tuning_v1`. This freezes
   96 development searches at 60s, requiring approximately 1.60 local worker-hours.
   Run `tune --batch ...`, audit the completed panel, and run `report-tuning` to
   retain its immutable `SELECTION.json`.
3. Collect 16 states per size separately for training and validation, using the
   passed recipe. Retain independent audits and complete input replay before
   `train`, which creates eight seeds per learned family. This is 176 states and
   at most 6,336 repair calls / 3.52 worker-hours plus overhead.
4. Use `evaluate --split validation --configuration .../SELECTION.json` with the
   complete checkpoint panel. All four methods run for 60s over five validation
   sizes, eight model blocks and ten starting seeds: 1,600 searches / 26.67 hours.
5. Only a completed, independently audited positive learning gate allows `freeze`.
   The subsequent held-out panel is 2,560 searches / 42.67 hours. Source, runtime,
   CPU identity, models and configuration must match the frozen contract.

The estimates exclude preprocessing, auditing and training, which must be recorded
separately. No paid resources have been provisioned. Reduce tuning breadth before
freezing if needed; do not reduce the declared final replication counts.

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

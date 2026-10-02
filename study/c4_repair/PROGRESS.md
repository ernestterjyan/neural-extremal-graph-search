# Progress

## 2026-10-02 — campaign started

- Started `research/c4-learned-repair` from GitHub main `55f2af2`.
- Read corrected study, parity follow-up and C4 pilot. Their results do not establish
  a competitive learned C4 solver. Frozen source and evidence remain untouched.
- Added separate graph operations, independent witness verification, prime-power
  polarity starts (trim / extend), shared region generation and CP-SAT incident-edge
  repair. Implementation has not yet passed validation or experimental gates.
- Held-out solver sizes have not been run. Paid expenditure: €0.
- Next: independent exhaustive small-graph tests; classical search; freeze and execute
  the 100-state feasibility protocol before training any region selector.

## 2026-10-02 — first implementation and failed feasibility gate

- Added multistart tabu, shared adaptive/random/learned neighborhood search, compact
  GNN and engineered-feature MLP rankers, supervised checkpoint training, immutable
  source/protocol snapshots, resumable collection/evaluation, and gated stage runner.
- The paired analysis averages searches within eight model-seed blocks and reports
  simultaneous 95% intervals against both designated comparators. Model loading,
  starting-graph initialization, features, inference, candidates and repair consume
  search time. Random streams for candidates, repair and diversification are separated
  from selector choice. Held-out evaluation requires a passed learning gate and freeze.
- **107 tests pass**, including exhaustive graph updates on all 1,024 five-vertex
  graphs, repair on all feasible four-vertex graphs and all nonempty regions, finite
  field axioms, permutation behavior, timed checkpoint accounting, checkpoint reload,
  block-level statistics and held-out protection. Lint and formatting pass.
- Six two-second classical smoke searches and two profile states are retained. They
  are calibration evidence, not a competitive solver comparison.
- `feasibility_v1`: **5/100 states improved — FAIL** against the predeclared 10/100
  threshold. n=35:1/50; n=40:4/50. No failed states. All 1,797 repair witnesses,
  initial states and stored starting pools independently verified C4-free.
- Regions: random / neighborhood / blocking path; sizes 3,4,5; two draws per family /
  size; one second per repair, each repairing the original strong state. Duplicate
  regions account for three omitted duplicate candidates. Development seed offset
  2,700,000. Every starting state is below its known exact optimum.
- Of 1,797 repairs, 527 have local-optimum certificates. Remaining outcomes do not
  prove that their regions admit no improvement. Local optima are not global bounds.
- Feasibility elapsed local wall time: about 0.400 h. Paid compute/storage: €0.
  `costs.jsonl` records retained experiment wall time and evidence hashes.
- No research selector has been trained; no held-out solver size has been run.
  The small checkpoint unit test is software verification, not research evidence.

### Next experiment

The relaxation sometimes spends its entire limit optimizing a graph that violates
C4 constraints. Test stopping the first invalid callback immediately, adding its
cuts, and continuing within the original deadline. Also tighten deadline bookkeeping
and retain event-level incumbent timing. This is a performance hypothesis, not an
established improvement. Preserve v1 unchanged; use a new profile and frozen gate
with fresh development seeds. Training remains blocked until a repair gate passes.

The complete campaign and manuscript are unfinished. Remaining work includes real
data collection/training, end-to-end validation, classical tuning, the primary freeze
and controlled comparison if gates pass, independent environment reproduction, cost
break-even where supported, and an outcome-matched manuscript. A failed gate alone
does not establish that all learned repair directions are infeasible.

## 2026-10-02 — repair revision and replay

- Changed separation to stop immediately on an invalid solver callback and add cuts
  within the remaining original deadline. Incumbent witnesses are now materialized
  before crediting their timestamps; new records retain improvement events.
- `profile_v3`: two strong states, regions 3/4/5, one-second repairs, no improvements.
  `profile_v4`: two fresh strong states, regions 3/5/7, three-second repairs, no
  improvements. These small probes neither prove infeasibility nor justify learning.
- Full suite still **107 passed** after the repair change. Seven campaign tests also
  passed after matching label/search configurations. Lint and formatting pass.
- Installed the v1 source and exact locked dependencies into a **fresh virtual
  environment**. Replayed a declared six-cell subset (first three locally certified
  cells at each size), preserving original regions/seeds/one-second limits. All six
  edge counts, local certificates, and complete witnesses matched; additionally
  verified by explicit C4 enumeration. This is the same physical machine, not an
  independent hardware replication or full timed study reproduction.
- `research/c4-learned-repair` has been pushed with the first failed gate retained.
- Next revision: freeze `feasibility_v2` with fresh seed offset 2,800,000, the revised
  engine, region sizes 3/4/5, four candidates per family/size, and two seconds per
  repair. At most 36 candidates per state (duplicates removed): <=2 local worker-hours
  plus overhead. Keep the original 10/100 threshold, strong state rule, all 100 states,
  all failure records, and independent witness verification. Do not train unless it
  passes. If it passes, labels and validation search must use this same repair recipe.

## Hypotheses and risks

1. Incident-edge regions can escape the polarity plateau under practical time limits.
   This must pass the feasibility gate before spending on learning.
2. Region quality is predictable and a selector benefits search after inference cost.
   Supervised ranking accuracy alone will not pass the learning gate.
3. Lazy constraint separation may spend its whole budget rejecting dense invalid
   relaxations. Log rounds/cuts/status and retain the original feasible graph.
4. Timed solver behavior varies with hardware. Freeze CPU concurrency and record
   initialization time, deadline overshoot, backend versions and source hashes.

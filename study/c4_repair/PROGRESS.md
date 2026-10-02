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

## 2026-10-02 — independent audit and input replay

- Added a separate standard-library auditor that imports no solver code. It checks
  common-neighbor feasibility, complete cell/seed panels, source snapshots, fixed
  complements, model identities, required search records and within-budget history.
- Re-audited all first-panel evidence: **5/100 FAIL**, 2,297 graph occurrences,
  527 claimed local certificates checked for status consistency. This is not an
  independent optimality-proof audit; v1 predates incumbent-event timing records.
- Replayed **all 100 v1 starting states and complete ordered candidate pools** from
  archived source and seeds. Preserved replay and auditor dependencies with hashes.
  Repair outcomes and timed performance were not rerun by this input-only check.
- **19 new integrity/replay tests pass**. They include real archived development
  inputs, a retained standalone replay, partial lines and deliberately omitted labels.
  Full solver tests are deferred until the timed job ends to avoid concurrent load.
- Identified the host as **Apple M5** through a permitted read and retained the
  measurement separately. Already-sealed v2 metadata stays unchanged; the primary
  freeze still needs a supported hardware-recording path.
- Documented an incorrect strengthening comment: adjacent selected vertices can
  still participate in C4s before lazy separation. Verified outputs remain feasible.
- The stricter audit also disclosed equal-score final repair witnesses that differ
  from the strict-improvement event graph. Frozen `accept()` permits this before the
  deadline. The event graph proves the timed gain; the replacement is verified and
  disclosed without assigning it the earlier graph's timestamp.
- v2 remains live with unchanged source/protocol. At the completed **71-state**
  audit: **22 improved** (n35:16, n40:6), no failures, 2,552 timed repair records,
  2,962 graph occurrences checked. This is a partial panel and does not authorize
  training. Numerical gate success alone will not establish escape from the best
  construction plateau or any learned-search advantage.
- Paid expenditure remains **€0**. Next: finish all 100 v2 states, retain its complete
  independent audit/input replay and costs, then resolve the comparator, timing and
  freeze gaps in `FORMULATION_AUDIT.md` before collecting/training research models.

## 2026-10-02 — controls staged for integration after v2

- Prepared and tested family/size adaptive selection, balanced paired method order
  and a host-wide process lease. Eight tests include real process exclusion and
  recovery from a terminated owner. The live runner has not yet adopted these controls.
- Added audited descriptive reporting for partial/failed/invalid evidence. Retained
  the full v1 report, including its construction-plateau comparison. Four tests pass.
- Confirmed an expressivity blind spot in the current unmarked GNN on C8, and added
  a candidate-membership GNN revision with three passing permutation/gradient tests.
  This establishes model behavior, not improved repair or search performance.
- `DEVELOPMENT_CONTROLS.md` records the integration and classical-tuning requirements.
  Rechecked the primary references and preserved the C4-only/girth-five distinction.
- At 97 completed v2 states: 28 improvements (n35:16, n40:12), no failures. The
  original source and all held-out sizes remain untouched. Paid expenditure: €0.

## 2026-10-02 — complete revised feasibility gate

- The confirmed live process finished normally. `feasibility_v2`: **28/100 PASS**
  against the unchanged 10/100 threshold; n35:16/50 and n40:12/50; no failed states.
- Independently checked **4,174 graph occurrences**, including all **3,593 timed
  repair records**, and 1,268 local-optimum/status claims. Optimality certificates
  are not independently proved. Disclosed 177 equal-score final-witness replacements
  without their own strict-improvement event; timed gains use recorded event witnesses.
- Replayed **all 100 starting states and complete ordered candidate pools** from
  the archived v2 source and seed schedule. Source/protocol stayed unchanged until
  the process exited and complete verification finished.
- Repairs: 3,514 zero gains, 77 one-edge gains, two two-edge gains. Best n35 graph:
  **102 edges**; best n40 graph: **125 edges**. Neither exceeds the strongest observed
  initial construction at that size. Known exact values remain 106 and 127.
- This meets the declared repair-availability gate, but demonstrates recovery of
  weaker starts rather than escape from the best construction plateau. No learned
  advantage, held-out transfer, or record has been established.
- Retained the full runner report, independent audit, input replay and descriptive
  report with source dependencies. Measured state execution totals **1.441 local
  hours**, recorded in `costs.jsonl`. Paid expenditure remains **€0**.
- After the timed process ended, the full software suite passed: **141 tests**,
  including the new controls, model and evidence checks. Lint/formatting pass. The
  campaign's feasibility guard also accepts the independently checked completed panel.
- Next: integrate/test the staged controls and GNN, tune strong classical search on
  development data, collect labels with sizes 3/4/5, four draws/family/size and two
  seconds/repair, then train eight independent models per family. Any functional
  repair-machinery change needs fresh gated evidence. The full campaign is unfinished.

## 2026-10-02 — integrate controls before classical tuning

- Integrated family/size adaptive selection, balanced method order and host-wide
  experiment exclusion into the production runner; shared repair machinery is
  byte-identical to the passed v2 source. The complete independently audited gate
  and retained input replay are still accepted.
- Integrated the marked-region GNN with architecture-tagged checkpoints. Loading
  research panels rejects changes to model/training source. No research model has
  yet been trained; expressivity and gradient tests are not performance results.
- Added timed construction and perturb/refill calibration, retaining the original
  pilot. An actual timed test confirms matching start pools, feasible witnesses,
  and within-budget event credit across both calibrations and both comparators.
- Implemented paired classical tuning (96 searches, four recipes per family,
  n25/31/35/40, three seeds, 60s), complete-evidence parameter selection and source
  compatibility checks. Its outcomes have not yet been observed or selected.
- Production training now requires complete independent label audit/input replay;
  evaluation freeze requires a complete independent validation audit. MLP outcomes
  are reported as secondary controls and cannot replace the primary GNN result.
- **151 tests pass**, including tenure behavior, panel completeness, retained
  parameter-selection tampering, and real process exclusion. Lint/formatting pass.
- Next: profile current selector overhead, independently audit a small timed
  calibration/classical workload, commit/push integration, then freeze and execute
  the full development tuning panel. Evaluation sizes remain untouched. Paid €0.

## 2026-10-02 — profile and timed calibration smoke

- Profiled untrained GNN/MLP selectors on n20/35/40, 36 regions, twenty scored
  repetitions per family. GNN median feature-plus-inference overhead was 1.194,
  1.447 and 1.541 ms respectively; MLP 0.752, 0.906 and 0.923 ms. These random
  models provide timing evidence only. Cold checkpoint loading remains unmeasured.
- Ran all four classical/calibration methods for two seconds on n20/35/40 with
  identical pools. Twelve completed, no failures; independently audited 93 graph
  occurrences and timing history. Tabu attained n35:106 at 1.174s; explicit cycle
  enumeration additionally passed. It matches a known optimum, not a new record.
- One seed per size cannot establish general baseline strength. Retained all
  outcomes: n20 all46; n35 tabu106/others102; n40 construction125/others124.
- Both new experiment contracts identify the actual Apple M5 CPU as verified.
  Retained complete audits/programs, source snapshots, reports and costs (€0).
- Next: commit integration and profile evidence, freeze tuning_v1 before outcomes,
  then execute its 96 paired sixty-second searches sequentially. No research models
  or held-out experiments have been run; the full campaign remains active.

## 2026-10-02 — freeze classical development tuning

- Frozen tuning_v1 before observing any outcome: 96 paired searches, n25/31/35/40,
  three seeds, four tabu and four adaptive recipes, 60s each. Source SHA
  `7f9b5d259e057f6d1ba8d029e2fd4d91eb4426413bfc7433a3c024af96dde312`.
- All methods use the same passed repair recipe and identical initial pools.
  Trial order is seeded independently of scores. Selection and tie-breaking are
  retained in the contract; no unsuccessful trial may be silently omitted.
- Runtime retains verified Apple M5 identity and exact package versions. Projected
  local search time is 1.60h plus overhead; no paid resources have been provisioned.
- Source will execute from its archived snapshot with the host-wide lease, retaining
  append-only cells and a run log. The protocol/source are committed before launch.
  Subsequent work must not change this sealed source or restart a live process.

## 2026-10-02 — launch frozen tuning

- Started the archived tuning_v1 runner, confirmed execution handle **22072** live.
  Source/protocol were committed before launch (e6a85dc); append-only results and
  run.log are retained. ACTIVE_RUN.json records the handle, not a claim that its
  owner JSON alone proves liveness. Poll the same handle; never restart on timeout.
- No concurrent solver, training, or test workload is permitted during the timed
  run. Useful reporting/input-replay development may proceed without changing the
  archived source. No parameter selection or learned benefit is claimed yet.

## 2026-10-02 — input integrity while tuning continues

- Confirmed archived-source tuning execution handle 22072 remains live; no restart.
  Retained DEVELOPMENT_PREFIX_1.json/md with its exact reporting/auditor programs:
  16/96 cells complete, no failures, 119 graph occurrences checked. Partial results
  do not authorize selecting parameters or establishing a learned advantage.
- Extended the input replay checker to complete timed starting pools and selector
  profile candidate pools. Its scope explicitly excludes timed trajectories and
  performance reproduction. The production guard now checks actual cell identities
  and completion, including omitted/duplicated-cell rejection. Six new tests are
  written; execution is deferred until the timed worker finishes. Lint/format pass.
- Tightened the future validation/evaluation freeze to require verified hardware
  identity before validation, unchanged hardware when freezing, and complete input
  replay. These checks do not alter the frozen tuning runner or repair machinery.
- Added retained classical configuration loading for the calibration stage. Planned
  after tuning: 36 paired development calibration searches at 60s, using identical
  sizes/seeds and the selected shared recipe (0.60 local hours, €0). The two-second
  smoke is useful software evidence but does not measure sixty-second performance.
- Source changes and pending tests remain separate from the archived live job.
  The complete tuning panel must finish and be audited before parameter selection,
  labels/training and end-to-end validation. Held-out sizes remain untouched.

## 2026-10-02 — bind checkpoint panel identities

- Reviewed checkpoint provenance while the archived tuning worker remained live.
  Added checks binding each binary checkpoint header to its declared family,
  architecture and training seed; declared seeds must be eight distinct integers.
  Matching a byte hash alone does not establish model/seed identity.
- Added five tests using explicitly untrained binary fixtures: valid header binding,
  hash-matching files with wrong seed/family/architecture, and duplicate declared
  seed rejection. They establish no trained model or search-performance result.
- Eleven added test cases now await the end of timed tuning; lint/formatting pass.
  No local test or second solver workload has run concurrently with it. The seven
  repair/classical/tuning source files remain byte-identical to the frozen contract.
- GitHub confirms the earlier research branch PR11 was merged. Its passing CI
  covered commit 177f974, not these later changes. No new CI or paid resource was
  launched to avoid making that old success stand in for pending verification.

## 2026-10-02 — second independent tuning prefix

- Retained DEVELOPMENT_PREFIX_2 with exact raw bytes, reporting program and auditor:
  **68/96 completed cells**, **538 graph occurrences**, no failures. The source and
  parameter-choice rule remain unchanged. This is still an incomplete panel.
- Confirmed execution handle 22072 still running at 15:12:19 UTC; the live journal
  had subsequently advanced into n40 trials. No restart, no parallel solver/test job,
  no parameter selection, no trained research model and no held-out search.
- Eleven new provenance/replay test cases remain pending until the worker is terminal.
  Both retained prefixes and pending code are committed separately from the live
  append-only journal. Paid local campaign expenditure remains €0.

## Hypotheses and risks

1. Incident-edge regions can escape the polarity plateau under practical time limits.
   This must pass the feasibility gate before spending on learning.
2. Region quality is predictable and a selector benefits search after inference cost.
   Supervised ranking accuracy alone will not pass the learning gate.
3. Lazy constraint separation may spend its whole budget rejecting dense invalid
   relaxations. Log rounds/cuts/status and retain the original feasible graph.
4. Timed solver behavior varies with hardware. Freeze CPU concurrency and record
   initialization time, deadline overshoot, backend versions and source hashes.

## 2026-10-02 — complete classical development tuning

- All **96/96** frozen tuning cells completed. The original execution handle had
  expired when revisited; retained lock metadata reports normal release and an OS
  lock probe confirmed no worker remained. No run was restarted.
- The independent full audit verified **745 graph occurrences**, no failed cells.
  All 96 complete initial starting pools replayed from the archived source. This
  replay does not reproduce timed search trajectories or elapsed performance.
- All **162 tests pass**, including the eleven deferred provenance/input-replay
  cases; repository-wide lint/format and diff checks pass. The passed feasibility
  evidence still satisfies the strengthened guards.
- The predeclared selection rule chooses **tabu/reactive_h5** (mean 96.000 edges
  at 60s, tied with fixed_h5 but ahead at earlier checkpoints) and
  **adaptive/explore_stagnation10** (mean 94.583; remaining tie resolved by ID).
  These means pool n25/31/35/40 and describe tuning only, not unseen performance.
- Retained all eight recipes, seed-level graphs, complete audit/replay sources,
  SELECTION.json and TUNING_REPORT.md. Results SHA:
  `0c7f97ee3a804ea4a27ef4829876afab4bec42996fd37bb954ce00907d0eeb33`.
- Search time was 1.600 local hours, paid expenditure €0. Next: sixty-second
  calibration using the selected recipe on the same development sizes/seeds,
  then real label collection, eight-seed training and end-to-end validation.
  No trained research model, learned benefit, held-out search or new bound yet.

## 2026-10-02 — resumable development calibration

- Split preparation from execution for the next timed calibration. Its source,
  configuration, sizes and seeds can now be committed before any outcomes, and a
  stopped runner can resume only on the same runtime/source/protocol. Completed
  and failed cells stay append-only; failures are never silently retried.
- Four meaningful resume checks pass: preserve a completed prefix, retain failed
  cells, reject changed runtime and reject duplicate cells. Full suite: **166
  passed**; repository-wide lint and diff checks pass. Classical search, candidate
  pool and repair engine remain unchanged from the completed tuning.

## 2026-10-02 — launch development calibration

- Committed/pushed calibration source and protocol before outcomes (8d1cd24).
  Running its archived source through verified execution handle **77296**: 36
  paired sixty-second searches at n25/31/35/40, three seeds, random region
  selection / construction / simple repair, using the retained tuning recipe.
- Source SHA `661c9bd5623d06724372713a0a27a85622b2deb3639ece33d8580e7685918816`;
  protocol SHA `aad2262b1217b71fdc0941ee66fdf9feae09c9d3422e1725ec1165d83fc7159e`.
  One local Apple M5 worker, projected 0.60 hours, paid cost €0.
- No concurrent tests, model training or second solver workload. The recorded
  handle must be polled before any restart; complete verification precedes labels.

## 2026-10-02 — verified calibration and final-stage software

- Development calibration completed **36/36** cells (execution handle 77296
  exited 0): independent audit checked **267 graph occurrences**, no failures.
  All initial pools replayed and exactly match their corresponding tuning pools.
  Retained full source, results, log, audit/replay programs, summary and report.
- Mean final edges over n25/31/35/40: construction **95.000**, simple repair
  **94.917**, random region selection **94.583**. Tuned tabu had **96.000**
  and adaptive **94.583** on the paired development seeds. These are descriptive
  development results, not trained-model replication or held-out inference.
- Simple repair reaches 103 edges in two n35 draws, while construction stays 102;
  tabu already reaches known optimum 106. No new bound or learned benefit.
- Calibration used 0.600 local hours, €0 paid. Results SHA:
  `2d0d52b99c17f2c79e60c0c4790ed143b72a28dabb6ff2f995a9b57447a8ab6c`.
- Added the prospectively fixed twelve-cell reproduction subset to future
  controlled protocols, with a distinct locked-environment runner. It verifies
  unchanged source/models, charges loading to timed search, retains discrepancies
  and explicitly distinguishes fresh software environment from separate hardware.
- Added matched cached-selection profiling of all sixteen future checkpoints
  against adaptive choice and separate training/label/tuning cost reporting.
  Thirty decisions per search is a declared conditional inference-only scenario;
  zero/negative savings mean no finite recovery count. This cannot replace the
  end-to-end quality target or prove acceleration.
- **179 tests pass**, including thirteen new reproduction/cost cases. Their mock
  binaries are untrained software fixtures, not research models. Repository-wide
  lint/format/diff checks pass. Core repair/classical machinery remains unchanged.
- Next: actual labels, eight independent training seeds per learned family,
  matched trained inference profile, and the full end-to-end learning gate.
  Held-out sizes remain untouched; the campaign remains active.

## 2026-10-02 — freeze actual label/training sequence

- Prepared source/seed/recipe contracts before outcomes for **96 training states**
  and **80 validation-label states**. Both use source SHA
  `13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb`,
  which passed 179 software tests. Recipe remains r3/4/5, four draws, 2s per repair.
- Retained a single sequential launch script and its hash: collect/audit/replay
  each split; train eight independent seeds per family for 30 epochs; profile
  all actual checkpoints against adaptive selection; audit/replay and report
  costs. No end-to-end validation or held-out experiment is included in this job.
- Max nominal label budget is 3.52 local hours plus preprocessing, training and
  verification. Paid cost €0. Any failure stops subsequent stages and is retained.
  Do not restart the whole script after interruption: inspect the handle and
  resume the actual incomplete stage using its archived source.

## 2026-10-02 — launch real labels and training

- Frozen label contracts and sequential plan committed/pushed before outcomes
  (f624693). Started and confirmed execution handle **60924** live. Current
  phase is training-label collection; later phases proceed sequentially only
  after complete independent verification. All execution uses the archived source.
- Temporary idle-sleep prevention is tied to the job lifetime; no permanent OS
  preference or paid resource was changed. Expected label states: 96 train / 80
  validation; later sixteen actual models and matched overhead/cost evidence.
- No end-to-end validation or held-out search is included in this running sequence.
  Poll its live handle rather than treating ACTIVE_RUN metadata as liveness proof.
  No concurrent tests or second experiment/training process. Goal remains active.

## 2026-10-02 — first actual label-quality evidence

- Confirmed sequence handle **60924** live; the training collection advanced
  through n20 states. It was not restarted, and no concurrent solver/test ran.
- Retained an immutable **14/96 training-state prefix**, separately from the live
  journal. Independent set-based audit checked **987 graph occurrences** and
  **502 repair outcomes**, with no failures; all 14 generated states and complete
  ordered candidate pools replayed from the archived source.
- **242 outcomes improve their original state**, and **8/14 states have differing
  region gains**, providing actual ranking signal. This is partial n20 data, not
  a trained selector or an end-to-end performance result. State-mix stratification
  is retained; improvements to weak starts must not be confused with improving
  the strongest construction. Seventeen equal-score replacements have no strict
  gain event and are disclosed by the audit. Solver-local optimal flags are not
  independently certified global optima.
- Prefix results SHA
  `10f7ca07dbe647dc60f2b915a4bc80ef09c6be7eb03462db6982002edb4b9ce3`;
  source remains `13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb`.
- Keep the fixed 96/80 labels, eight seeds per family and later validation panel.
  Partial-label costs are not double-counted; the sequence books completed stage
  costs once. Paid resources remain €0; held-out sizes remain untouched.

## 2026-10-02 — predeclare full validation execution

- Confirmed real-data sequence handle **60924** live. Training labels have 72/96 retained states with 0 recorded failures at this observation; this is still incomplete data, not a trained-model result.
- Prepared the next launch script and its immutable execution plan before validation outcomes: all four methods, five validation sizes, eight model blocks, ten paired search seeds, 60 seconds each (**1,600 cells / 26.67 nominal hours**). No held-out or record search is included.
- Launch requires an authoritative successful exit and complete reviewed evidence from the current sequence. Preflight checks real labels, all sixteen checkpoints, matched profiling, source and cost bindings. The runner freezes the full source/models/runtime contract before its first search; post-run stages audit, replay, analyze and book costs.
- Shell syntax and both embedded Python blocks parse; all current source-manifest bytes remain identical to the passed 179-test source snapshot. No tests or second solver/training process ran alongside the live worker. This execution wrapper is not evidence that the validation gate passed.
- A failed learning gate leads to the declared distinct-environment reproduction and negative report. A passed gate permits the later held-out freeze. Paid spend remains €0.

## 2026-10-02 — complete verified training-label dataset

- All **96/96 training states** completed with no failure. Independent set-based audit checked **3,445 repair outcomes / 5,950 graph occurrences**; all 96 generated starts and complete ordered region pools replayed from the frozen source. Retained program hashes, audit/replay bindings and the full expected seed/state panel were checked.
- **1,115 positive repair outcomes** and **49/96 states with differing region gains** provide real supervised ranking signal. Stratification: best-of-pool 4/24 informative states, trim 0/18, extend 9/18, uniform 18/18 and perturbed-best 18/18. Improvements predominantly recover weaker starts.
- Best repaired edge counts equal the best initial counts at every training size: n20:46, n24:59, n28:76, n32:92, n36:106, n40:125. Improvements in four best-of-pool draws at n36/n40 do not exceed the strongest initial graph elsewhere in this dataset. No learned advantage, transfer result or new bound follows.
- The audit checks 1,432 solver-local optimal status flags for consistency, not independent optimality certificates. It discloses 122 equal-score replacements lacking a strict-gain event. Raw witnesses, gain events, audits/replay programs, logs and stratified `LABEL_QC.json` are retained.
- Results SHA `b35d4010ffd185626f1f5ed0b14b2a3173e4aa062955cfea6c9b0f2cdad185ea`; protocol SHA `4f7fe70e50383d7f59dd943c31be0a1a47c3ef65017e54b76c26acea56198e51`; source remains `13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb`.
- Collection used 1.321 local worker-hours. The full sequential job books its completed stages once later; these costs were not added twice. Paid resources remain €0. Confirmed execution handle **60924** live and advanced to **validation-label collection**. Actual model training and controlled validation still remain.

## 2026-10-02 — draft methods and interpret verified development results

- Added a manuscript draft covering the repair formulation, a conditional optimality argument for feasible optimal relaxations, polarity feasibility, implemented selectors, classical controls, fixed replication/analysis, completed evidence and pending results. It makes no trained, transfer, record or campaign-completion claim.
- Rechecked primary references. The strongest training starts at n20/24/28/32 equal known exact values 46/59/76/92, so this portion of the strong-start panel is mathematically saturated. Known development values at n36/n40 are 110/127; their retained best starts 106/125 remain below them. This contextualizes weak-state ranking signal without changing frozen methods or success criteria. No reference witnesses were downloaded.
- Retained a draft evidence manifest binding its completed data, audited counts, source hashes and reference checks. Checked all training-stratum table values and local evidence links. Source remains the unchanged 179-test snapshot; no CPU-heavy tests, second solver or training process ran alongside the live worker.
- The active sequence is collecting validation labels; actual checkpoints, overhead profile, controlled validation, conditional held-out evaluation and final reproduction/report still remain. The draft is useful work toward the final deliverable, not completion.

## 2026-10-02 — validation labels reach final two size groups

- Reconfirmed sequence handle **60924** live. Validation supervision has **50/80 completed records** and **0 recorded failures** at this observation. The first three groups, n22/n26/n30, each contain all sixteen declared states; collection has reached n34.
- Updated the durable worker handoff with the current collection phase and an explicitly dated snapshot. This is not a completed validation-label audit or end-to-end performance result. Source and declared counts are unchanged. No tests, second solver or model training ran alongside collection.
- Continue this live sequence through n34/n38 labels, full independent audit/input replay, all sixteen actual model trainings and matched profiling/costs. Then review its terminal exit and completed evidence before launching the separately declared 1,600-search validation comparison. The larger held-out panel remains untouched and gated.

## 2026-10-02 — complete real labels, models and matched overhead

- Sequence handle **60924 exited with code 0**. All 80/80 validation supervision states completed with no failure. Independent set-based audit checked **2,859 repair outcomes / 4,642 graph occurrences**; all generated starts and complete ordered region pools replayed. The expected size/seed/state panel and retained auditor/replay/source hashes were checked.
- Validation labels contain **863 positive outcomes / 42 informative ranking states**. Best-of-pool 0/20 informative, trim 5/15, extend 7/15, uniform 15/15 and perturbed-best 15/15. No repaired graph exceeds its size's strongest initial count: n22:52, n26:67, n30:85, n34:99, n38:116. The audit discloses 123 equal-score replacements lacking a strict-gain event and 1,176 solver-local optimal status flags checked for consistency, not independent optimality proofs.
- Validation results SHA `feff8fcbd2bc8e5a68b3448d29e121d4aa44864e82323c277a9c01b008a5a353`; protocol SHA `2d6f45dbc0a9d948c30d4461b55fa4ce5e529d6eee4d9fa493f2e745f27d9d97`; source remains the unchanged 179-test snapshot `13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb`.
- Trained all **eight GNN and eight MLP seeds** 6000000–6000007 on real data, thirty epochs each. All epoch losses are finite, best validation epochs match retained checkpoints, metadata/data/source bindings and checkpoint headers/hashes pass. Retained best/last weights, full epoch records and `TRAINING_REVIEW_v1.json`. Actual model training totals 48.278 GNN seconds plus 11.563 MLP seconds; validation losses are descriptive, not a solver-performance result.
- Trained overhead profile completed all seventeen conditions on three declared development graphs; independent witness audit and full input replay pass. GNN mean graph-median selection latency is 1.379–1.391 ms, MLP 0.866–0.878 ms, adaptive 0.004535 ms. All learned conditions are slower, so no finite inference-only recovery count is demonstrated. Quality under the full charged search budget remains the decisive test.
- The sequence booked all four completed stages exactly once. Total retained local worker time is **6.508 hours**, paid compute/storage **€0**. Cost report hashes match models, profile and ledger. Reviewed the full independent audits/replays with no concurrent solver or test job.
- Next: commit/push completed evidence, then launch the predeclared **1,600-search end-to-end validation panel** sequentially. A trained model, supervised label improvement or profiling result does not pass the learning gate. Unseen evaluation and record search remain unrun and gated. The goal remains active.

## 2026-10-02 — launch complete end-to-end validation

- Completed actual labels, sixteen checkpoints, matched profiling and costs committed/pushed as **b6572e7**. All preflight source/model/data/audit/replay/cost bindings pass. No test or second campaign worker runs concurrently.
- Launched the unchanged **1,600-search validation panel** with verified Apple M5 identity, one CPU worker and temporary idle-sleep prevention tied to its lifetime. Confirmed execution handle **98945** live and its first GNN search at n22 completed. This is startup evidence, not a learning-gate result.
- The runner retained source, models, environment and the complete protocol before its first search. Validation protocol SHA `84a4479fbc241a8dfd9b964c527dcc3229097c3281c5b7979428767f80e68596`; source remains `13dfb0462b19f14913184fc98c10d234c6c66ea247d9efb1dc4899b66b16eafb`. Five development sizes × eight blocks × ten starts × four methods × sixty seconds require 26.67 nominal search-hours, plus overhead.
- Continue through the full panel, independent audit, deterministic input replay, paired report and unique cost booking. A failed learning gate stops held-out/record search and requires the declared distinct-environment reproduction and negative manuscript. A passed gate permits the subsequent final freeze. Do not infer from the partial journal, reduce replication or restart the whole launch script after interruption. The goal remains active; paid compute/storage remains €0.

## 2026-10-02 — first complete validation size/block group

- Reconfirmed execution handle **98945** live. The first block at n22 completed all **forty searches**: four methods over ten paired starting/search seeds, with no recorded failure. Each logged final count is 52. This is one size in one block, not the full learning-gate panel or an independently audited final result.
- Full validation remains **1,600 searches**, across five sizes and eight model blocks. Retained working journals and unchanged source/protocol are continuing sequentially; the post-run independent witness audit, full input replay, paired analysis and costs still remain. No held-out search, concurrent heavy work, protocol change or paid resource was introduced.

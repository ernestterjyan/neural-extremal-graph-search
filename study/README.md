# Corrected transfer study

This directory is separate from the published v0.1/v0.2 artifacts. Read `RESEARCH_REPORT.md` for the assessment, `protocol.json` for the prospectively fixed protocol, `MATHEMATICS.md` for diagnostic definitions, and `RELATED_WORK.md` for literature positioning. `ledger.jsonl` retains execution events, including recoverable orchestration errors.

## Source and environment

The study starts from clean commit `87a6bc5a52f973b8025bdd5efeb0791fcc4dcb01`. The training implementation and frozen protocol are committed as `4b14e4b` on `research/corrected-transfer-study`. The original checkout and its historical artifacts were preserved. New work lives in this isolated clone.

Canonical runs use Python 3.12, CPU, deterministic PyTorch operations and one intra-operation thread per worker. Four independent worker processes are allowed. The exact installed package versions, platform, thread settings, source hashes, dirty status, configurations and source archives are retained in every training run. A shared environment can be reused locally, but each launch imports this checkout's `src`; it does not import the original checkout's source.

The dependency specification and `uv.lock` travel with each source snapshot. On a new machine, use `uv sync --locked --extra dev`. Reproducing identical trajectories requires the recorded numerical environment. Different PyTorch/BLAS/device implementations, floating-point rounding, or hardware may change near-tied action probabilities. Independent episode RNG streams remove batching-induced random-stream coupling; they cannot promise cross-platform bitwise neural arithmetic.

## Regenerate the study

Commands below run from the repository root, with the project's Python 3.12 environment. The experiment scripts insert this checkout's `src` explicitly.

```sh
uv run python experiments/run_study.py all
uv run python experiments/analyze_study.py
uv run python experiments/exact_diagnostics.py
```

Individual stages are `train`, `evaluate`, `budget`, and `verify`. A stage resumes completed compatible cells and training checkpoints. Training is limited to five seeds for four families and 100 iterations per curriculum stage. Main evaluation covers 87,000 graphs and the fixed-budget comparison another 4,000. The fixed-budget checkpoints are current models after 50 iterations (12,800 training episodes), before stage restoration; action counts, updates and size exposure still differ.

To regenerate tables and figures from the evidence bundle without retraining:

```sh
uv run python experiments/run_study.py verify
uv run python experiments/analyze_study.py
```

Verification fails if any planned main/budget cell is missing or incomplete, if a chunk/hash is damaged, or if a saved graph is invalid. Analysis requires the verification record, both complete CSVs, and all 20 training logs/checkpoints. Missing training inputs cause an error; figures are not silently replaced by incomplete plots. The separate historical reporting command warns explicitly if old training logs are missing.

## Evidence bundle

Large generated evidence lives under `study/artifacts/`, excluded from ordinary source control. It includes full training logs, final/stage/fixed-budget weights, optimizer resume checkpoints, source snapshots, all final evaluation graphs and complete action sequences. A checksummed `study/evidence-corrected-v1.tar.gz` is the portable handoff; see `bundle-manifest.json` for its hash and contents. It is local and has not been uploaded or published. Transfer that file alongside a clean source checkout, verify its checksum, and extract it at the repository root. Do not regenerate artifacts into a different implementation and silently reuse old manifests.

Main graph chunks are under `study/artifacts/evaluation/<method>/seed-<s>/n-<n>/`. Each compressed JSON chunk stores a contract hash, stable result identifiers, graph hashes, sorted final edge lists, action sequences, structural outcomes and (for the declared subset) decision diagnostics. Reconstruct any state by taking an action-sequence prefix. `tables/stratified_examples.csv` indexes one example for each observed method/size/seed/outcome stratum. Absent strata are not fabricated. Fixed-budget results are stored separately under `study/artifacts/budget/`; never pool their identities with main rows.

The separate evidence archive and source checkout are both required for a complete reproduction. Historical full training records recovered from the original local runs are retained under `study/artifacts/historical-training/` with `historical-training-manifest.json`. Their original missing metadata fields remain missing; these are archived records, not repaired retroactively.

## Evaluate the included historical release directly

No ignored original `runs/` directory is needed:

```sh
uv run python experiments/reproduce_release.py
uv run python experiments/reproduce_release.py --full
```

`study/release-evaluation.json` points directly at the ten committed release weights. The default replays the first 64 n=24 episodes for each GNN/MLP seed (640 total), checks every persisted graph independently, and compares every edge count against the published CSV. `--full` replays all 200 episodes at the eight original sizes for both trained families. These are deterministic historical replays, not independent new observations.

Legacy sampling is explicitly named `legacy-batch-v1` and requires the historical batch size and grouping. Corrected training/evaluation use `episode-v2`. Do not use the old evaluation CLI with changed default RNG semantics and expect historical exact replay.

## Resume and failure contract

A new training launch cannot overwrite an existing run. Resume requires the resolved training contract (including learning rate, model, seed, curriculum and sampling protocol) to match; run name/output relocation are excluded from the scientific contract, but the original run records must accompany the checkpoint. Configuration/optimizer compatibility is checked before run records are modified. Python, NumPy, PyTorch and selection RNG states are restored. Launch metadata is preserved and resume events appended. Source/environment provenance is recorded for both launches and resumes; use the frozen source revision for canonical continuation.

Evaluation uses atomic chunks tied to source/config/checkpoint hashes. Completed episodes are not resampled on compatible resume. Numerical failures produce failure records and abort the affected cell; they do not become terminal graphs or ordinary low scores. Failed seeds are not silently replaced.

The test suite covers invalid numerical logits, true terminal masks, batch/order/chunk invariance, historical replay, actual interrupted training/evaluation, rejected optimizer changes, metadata preservation, exact completion annotations against independent balanced-cut enumeration, look-ahead against brute force, and corrupted evidence rejection.

## Interpretation boundaries

The environment supplies legality and informative handcrafted features. The direct Turán reference uses the known solution. The candidate-only and endpoint controls have fewer parameters and receive different information than the global MLP; equal hyperparameters do not guarantee equal optimization difficulty. Report all five seed outcomes, realized budgets, failed curriculum gates, and uncertainty. The old GNN/MLP comparison has a training-version confound and is kept as historical context only.

## Supplementary intervention and checks

`intervention-protocol.json` specifies an exploratory parity-constraint intervention before its new evaluation draws. It adds component parity to a frozen GNN and to a uniform policy, with a fresh unmodified-GNN arm. This explicitly extends the original intervention gate based on observed odd-cycle failures; it is not a primary confirmatory comparison or evidence that aliasing caused those failures. It uses five frozen GNNs, four sizes and 6,000 fresh episodes in a separate namespace.

```sh
uv run python experiments/parity_intervention.py
uv run python experiments/audit_study_evidence.py
uv run python experiments/check_trained_symmetry.py
uv run python experiments/regenerate_historical_reports.py
uv run python experiments/profile_inference.py
```

Run the serial inference profile only after other workers have stopped. It measures single-graph latency separately from batch throughput. The historical regeneration command requires all ten archived training logs and writes six figures into a separate directory, preserving the published originals.

A final-stage interruption regression was discovered after the frozen training cohort began. Its repair saves stage artifacts before the resume checkpoint advances, and retains recovery samples for interrupted finalization. The full seed-0 GNN repetition under this repair matched every loss, reward, validation value, action count, update count and final model parameter exactly. `boundary-repair-equivalence.json` records that check. The primary 20-run cohort uses the original common training snapshot; final evaluation uses the repaired source with unchanged inference code. Both snapshots and the protocol deviation are retained.

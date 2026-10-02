# Campaign runbook

All paths below are repository-relative. Install `uv sync --extra dev --extra repair`.
Invoke `.venv/bin/python` directly so optional solver dependencies remain installed.
Run timed experiments sequentially, on one CPU worker. Do not launch training or
another search concurrently with a timed experiment.

## 1. Verify and profile

```sh
.venv/bin/python -m pytest
.venv/bin/python experiments/c4_repair.py profile --output study/c4_repair/profile_NAME --seed 810000 --sizes 35 40 --region-sizes 3 5 7 --seconds 1
.venv/bin/python experiments/c4_repair.py verify --batch study/c4_repair/profile_NAME
```

Profiles are development evidence. Retain unsuccessful profiles. Never overwrite a
batch directory or silently reuse inspected draws as unseen evidence.

## 2. Freeze and run feasibility

```sh
.venv/bin/python experiments/c4_repair.py prepare --output study/c4_repair/feasibility_NAME --seed 2700000 --region-sizes 3 4 5 --per-family 2 --seconds 1
.venv/bin/python experiments/c4_repair.py feasibility --batch study/c4_repair/feasibility_NAME
.venv/bin/python experiments/c4_repair.py report --batch study/c4_repair/feasibility_NAME
```

The gate draws 50 states at each of n=35 and n=40. Every state is the strongest
graph in its independently generated starting pool, below the known exact optimum.
Each region repairs the original state independently; the entire pool is tested.
A state counts once regardless of how many regions improve it. Failed states count
as not improved and remain in the evidence. Pass requires at least 10 of 100 states.
If it fails, revise the repair neighborhoods/limits on development evidence and run
a separately frozen, clearly identified gate. Do not train around a failed gate.

Every batch contains its protocol, environment, source hashes and a source snapshot.
If interrupted without source changes, feasibility resumes completed cells. To resume
an old batch after changing source, use its snapshot with its own `src` on PYTHONPATH:

```sh
PYTHONPATH=study/c4_repair/feasibility_NAME/source/src .venv/bin/python study/c4_repair/feasibility_NAME/source/experiments/c4_repair.py feasibility --batch study/c4_repair/feasibility_NAME
```

## 3. Collect supervised outcomes and train

These commands are blocked until a verified feasibility gate passes. The example
data breadth is provisional development tuning; the final eight training seeds are
mandatory. Train and validation sizes cannot be interchanged.

Set collection `--seconds`, `--per-family` and `--region-sizes` to the exact values
that passed feasibility; set validation `--repair-seconds`, `--per-family` and
`--region-sizes` to that same recipe. The defaults below illustrate the first recipe,
which failed and therefore does **not** authorize training.

```sh
.venv/bin/python experiments/c4_repair.py collect --feasibility study/c4_repair/feasibility_NAME --split train --output study/c4_repair/labels_train_NAME --seed 4000000 --states-per-size 16
.venv/bin/python experiments/c4_repair.py collect --feasibility study/c4_repair/feasibility_NAME --split validation --output study/c4_repair/labels_validation_NAME --seed 5000000 --states-per-size 16
.venv/bin/python experiments/c4_repair.py verify --batch study/c4_repair/labels_train_NAME
.venv/bin/python experiments/c4_repair.py verify --batch study/c4_repair/labels_validation_NAME
.venv/bin/python experiments/c4_repair.py train --feasibility study/c4_repair/feasibility_NAME --data study/c4_repair/labels_train_NAME --validation-data study/c4_repair/labels_validation_NAME --output study/c4_repair/models_NAME --seed 6000000 --epochs 30
```

The label target is improvement actually obtained within the common repair limit,
not an unproved local optimum. The compact GNN and engineered-feature MLP both
rank the same candidate pools. Eight independent initialization/shuffling seeds
are used for each family. Common training data do not provide eight independent
datasets; the declared replicates are trained models.

## 4. End-to-end learning gate

```sh
.venv/bin/python experiments/c4_repair.py evaluate --split validation --feasibility study/c4_repair/feasibility_NAME --checkpoints study/c4_repair/models_NAME --output study/c4_repair/validation_NAME
.venv/bin/python experiments/c4_repair.py report --batch study/c4_repair/validation_NAME
```

This uses all five validation sizes, eight model-seed blocks, ten paired starts per
size/block, 60 seconds per search. Checkpoints at 1, 10, 60 seconds retain graph
witnesses. Model loading, graph construction, candidate generation, feature
extraction, inference and repair are charged. Calibration and tuning must be
finished before the final validation batch. The gate requires positive mean
GNN–adaptive advantage at 60 seconds; confidence intervals are also reported.
Ranking loss alone is not evidence of a useful search solver. If the gate fails,
stop the larger neural campaign and document the negative/classical result.

## 5. Freeze once, then evaluate

```sh
.venv/bin/python experiments/c4_repair.py freeze --validation study/c4_repair/validation_NAME --checkpoints study/c4_repair/models_NAME --output study/c4_repair/frozen_protocol.json
.venv/bin/python experiments/c4_repair.py evaluate --split evaluation --freeze study/c4_repair/frozen_protocol.json --checkpoints study/c4_repair/models_NAME --output study/c4_repair/evaluation_NAME
.venv/bin/python experiments/c4_repair.py report --batch study/c4_repair/evaluation_NAME
```

Commit the freeze before evaluation. Held-out arguments come from the frozen file,
not command-line overrides. Source and model changes invalidate it. The primary
panel is n=44,48,52,56,64,72,80,96; 640 searches per solver, four solvers, at least
42.67 CPU-worker hours of search budget. Validation adds at least 26.67 hours.
Data collection and tuning are additional. Project full cost from profiles before
requesting paid provisioning; do not reduce final replication to fit an estimate.

Primary inference averages ten searches per size within each training-seed block,
then compares the eight paired block means. Two Bonferroni-adjusted Student t
intervals have simultaneous 95% coverage under their assumptions. The mean gain
must be at least one edge and both interval lower bounds positive; GNN must beat
both comparators in mean on at least six of eight sizes. Secondary checkpoint
results never replace a failed primary outcome.

## 6. Record attempt and independent reproduction

Proceed to exploratory published-witness starts only after the primary passes.
Recheck the current literature, document which lower bound and date are exceeded,
and verify the candidate with the independent set-based common-neighbor checker
and explicit C4 enumeration. A new witness proves a lower bound, not exact optimality.

Reproduce a declared subset in a second environment using the source/lock snapshot.
Record deterministic work-limit replay separately from timed replay. Report all
failures, source/checkpoint hashes, seed-level outcomes, graph witnesses, paid and
local costs, and training/tuning break-even where supported. Write claims matching
the actual gate or primary outcome. No record, learning or transfer claims are
currently supported by this new campaign.

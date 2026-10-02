# Controls prepared during the live feasibility run

The following revisions are staged as separate development modules. They are tested
but **not yet integrated into the live runner or used in solver experiments**. The
frozen v2 source remains unchanged. Integrate and test them after that process ends,
then retain the exact methods in subsequent source snapshots.

## Adaptive neighborhood selector and experiment execution

`experiments/c4_repair_controls.py` provides:

- A non-neural adaptive selector over **family × actual region size**, with an
  explicit exploration floor. It updates from obtained repair gain per elapsed repair
  second. Repeated candidates in one arm do not change that arm's sampling probability.
  Region generation, repair, acceptance and diversification remain shared with learning.
- Seeded method-order permutations, rotated across model blocks. For four methods
  and eight blocks, every method appears twice in every chronological position for
  each size/starting-seed group. Search outcomes never influence the order.
- A host-wide POSIX process lease for all timed experiment and training stages. OS
  locks, not stale JSON files, enforce exclusion across checkouts. A process crash
  releases the lock; the file is never unlinked to avoid inode races.

Eight tests verify size adaptation, exploration, deterministic selection, paired
position balance, actual cross-process exclusion and recovery after abrupt termination.
These tests are software evidence, not end-to-end validation or baseline-strength evidence.

## Region-conditioned GNN

The current GNN embeds an unmarked graph once and then pools selected vertices.
On C8, R={0,2} and R={0,4} have identical per-node/engineered features and identical
unmarked node embeddings. The scores therefore coincide for every parameter choice,
despite different joint geometry. This does not prove different repair gains.

`experiments/c4_repair_region_gnn.py` adds a region-membership bit before message
passing and batches the candidate-specific marked graphs. Hidden width 32 and three
message-passing layers remain compact; the same twenty engineered features also
remain available to the MLP. Three tests check the geometry distinction, graph and
candidate permutation behavior, single/batched consistency and finite gradients.

Candidate conditioning costs more inference work than one shared unmarked embedding.
That cost must be profiled and charged during validation. No research model has been
trained and no learned benefit is claimed.

## Descriptive failure reports

`experiments/c4_repair_state_report.py` makes a separate independently audited report
even when a panel is unfinished or contains failed cells. Invalid evidence produces
an explicit audit-error report. Partial feasibility is PENDING regardless of its
current number of gains. Controlled descriptive means are labelled as potentially
unbalanced and never replace the frozen paired primary analysis. The MLP is shown
alongside the other methods whenever such a panel exists.

The retained `feasibility_v1/DESCRIPTIVE_STATE.json` and `.md` confirm 5/100 FAIL and
no state exceeding the strongest observed initial graph at either size. Four tests
check pending, failed and invalid evidence handling and the retained standalone
program. Each report retains its exact
program and auditor dependency.

## Classical tuning before the final validation batch

Use only development sizes. A proposed bounded panel is n=25,31,35,40, three paired
starting/search seeds per size, four configurations per classical family, sixty seconds
per search: 96 searches / 1.60 worker-hours, plus overhead. This is a design proposal,
not a frozen or executed experiment. Freeze the actual source, configurations and
seed schedule after integration and profiling, before running the tuning panel.

Include a fixed-penalty, fixed-history-five, restart-1000 tabu recipe inspired by
[Mehrabian et al., IJCAI 2024, sections 5–6](https://www.ijcai.org/proceedings/2024/0772.pdf),
as well as the current adaptive-penalty variant. Their objective forbids C3 and C4;
ours permits triangles, so remove the triangle penalty and describe the adaptation
honestly. This is not a reproduction of their distributed seven-day experiment.

Choose classical parameters using the paired sixty-second development outcomes,
then freeze them for the separate end-to-end validation and untouched evaluation.
Confirm tabu tenure means the declared number of subsequent banned moves; do not
retain an off-by-one interpretation as a hidden hyperparameter. Tune adaptive
exploration and shared stagnation controls with the same budget and record all trials.

The [Afzaly–McKay table](https://users.cecs.anu.edu.au/~bdm/data/extremal.html), rechecked
2026-10-02, gives exact C4-only values n25:63, n31:90, n35:106 and n40:127. Its graph
counts and edge counts are distinct; a ≥ on the graph count does not make the edge
count a lower bound. Do not download target-size witnesses for controlled starting pools.

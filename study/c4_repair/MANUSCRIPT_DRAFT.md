# Draft learned region repair for C4 free extremal graph search

## Abstract

We investigate whether a learned choice of repair region improves the search for
dense simple graphs without four-cycles. A common repair engine changes edges
incident to a small vertex set while preserving the remaining graph. A graph
neural network and an engineered-feature multilayer perceptron rank candidate
regions using gains measured by the same time-limited solver. Adaptive region
selection and multistart tabu search provide classical comparators. The completed
development work establishes a feasible repair implementation and genuine
supervised ranking signal, but does not establish an advantage from learning.
The revised feasibility panel improved 28 of 100 states; the complete training
dataset contains 3,445 repair outcomes and 49 of 96 states with differing region
gains. In both panels, the strongest repaired graph at each size matches the
strongest initial graph in that panel. Controlled validation, transfer evaluation
and final trained-model reproduction remain incomplete. This draft reports the
implemented methods, the prospective comparison and the current evidence.

## Research question and related work

For a fixed number of vertices, the objective is to maximize the edge count of a
simple undirected graph containing no four-cycle as a subgraph. Triangles are
allowed. A verified graph gives a lower bound on the extremal edge count; finding
such a graph does not prove that a denser graph is impossible.

The proposed contribution is a useful search decision: choosing where to rebuild
an existing graph. Its value must survive comparisons that include the costs of
features, inference, initialization and repair. A favorable training loss alone
cannot answer this question. Improvement over adaptive selection would support
the value of learning; the separate MLP control tests whether graph message
passing adds value beyond engineered features.

Mehrabian and colleagues study neural and tabu search for graphs excluding both
triangles and four-cycles. Their incremental search uses strong smaller graphs to
initialize larger ones. Their tabu experiments use a much longer, parallel search
campaign. This motivates a strong classical comparison here, but the different
forbidden subgraphs, budgets and implementations prevent a claim that our local
tabu comparator reproduces their published results or exhausts classical search
methods. [Finding Increasingly Large Extremal Graphs with AlphaZero and Tabu
Search](https://www.ijcai.org/proceedings/2024/0772.pdf).

## Mathematical formulation and verification

Let `a_uv` be a binary edge indicator for each unordered pair of distinct vertices.
The full problem maximizes `sum(a_uv)` subject to a constraint
`a_uv + a_vw + a_wx + a_xu <= 3` for every cyclic ordering of four distinct
vertices. Chords do not remove the forbidden cycle. For a repair region `R`, the
indicators with both endpoints outside `R` remain equal to the input graph; all
indicators with at least one endpoint in `R` may change. The objective also
requires a result with at least the input edge count, so the original feasible
graph remains an available incumbent.

A simple graph has no four-cycle if and only if every pair of distinct vertices
has at most one common neighbor. Two distinct common neighbors supply a
four-cycle, and opposite vertices of any four-cycle have two common neighbors.
This equivalence supports a set-based witness verifier independent of the solver
and the bitset graph-update code. Future record candidates additionally require
explicit four-cycle enumeration.

The repair model starts with valid strengthening constraints derived from the
fixed complement and adds violated cycle inequalities as they are discovered.
For a discovered cycle, its variable-edge sum is bounded by three minus its
fixed-edge count. All fully feasible repairs satisfy these cuts. The remaining
model is therefore a relaxation of the full repair problem. If its proven optimal
solution is also four-cycle free, its edge count bounds and attains the full
repair optimum. This certifies optimality conditional on the fixed complement,
not global optimality of the extremal graph problem for a proper vertex region.
This argument does not depend on static strengthening covering every cycle with
two region vertices; the implementation relies on subsequent separation for
uncovered cycles.

One CP-SAT worker uses the original repair deadline across successive rounds.
Only verified feasible incumbents are retained; a timeout returns such an
incumbent. Solver statuses retain their distinct meanings. A feasible solution
without an optimal status is not an optimality certificate, and an optimal
relaxation containing a four-cycle is not yet a feasible repair. The independent
evidence checker validates witnesses and status consistency; it does not check
an independent CP-SAT proof certificate. [CP-SAT status documentation](https://developers.google.com/optimization/cp/cp_solver).

## Starting graphs and common search mechanisms

For a supported prime power `q`, the construction uses the projective points of a
three-dimensional vector space over a finite field. Distinct points are adjacent
when their representatives have zero dot product. The dot product is
nondegenerate. Two distinct projective points give independent linear forms,
whose common kernel is one-dimensional. They consequently have at most one
common orthogonal projective point. Removing loops preserves this property and
therefore produces a four-cycle-free graph. This is the rationale for using
polarity constructions; it is not claimed as a new construction.

The implemented field panel is `q = 2, 3, 4, 5, 7, 8, 9, 11`. A starting pool
trims a nearby larger construction, extends a nearby smaller construction, and
adds greedy constructions. Generated starts are shared across competing methods
through paired seeds. Published witnesses at evaluation sizes are excluded from
training and controlled starting pools.

Candidate regions combine random vertex sets, local neighborhoods and vertices
on paths blocking a missing edge. The retained development recipe uses region
sizes three, four and five, with four draws per family and size and a two-second
repair limit. Deduplication can reduce the number of candidates in a state.
GNN, MLP and adaptive selection use the same candidate generator, repair engine,
acceptance behavior and diversification rule. After ten stagnant repairs, the
trajectory restarts or perturbs a graph by deleting edges incident to three
vertices and greedily rebuilding; the best feasible incumbent is preserved.

The separate tabu comparator flips edges and can traverse graphs with
four-cycles, using a penalty to guide the trajectory while retaining only
feasible reported incumbents. Development tuning selected tenure five, an
adaptive penalty and restart after 1,000 stagnant iterations. Adaptive region
selection uses region-family and size arms, 40% exploration and smoothed rewards
based on realized gain and repair duration. Both designated classical choices
were selected from retained development trials before controlled results.

## Supervised region selectors

Each labeled state evaluates every candidate region independently on the same
original graph. Targets are realized improvements under the declared repair
limits, rather than unknown exact gains for every region. Repair seeds and
incumbent histories remain part of the evidence. A target distribution is formed
by applying softmax to the gains; training minimizes its cross-entropy against
the softmax of predicted region scores. States with equal gains have a uniform
target and provide no observed preference among regions.

The MLP uses twenty engineered region features and two hidden layers of width 64.
Features include graph and region size, normalized edge density, degree
statistics, internal and boundary density, triangle incidence, two-hop reach,
legal edge incidence and region family. The GNN marks membership of each
candidate region in addition to five node features, applies three message-passing
layers of width 32, and combines selected-node mean and maximum pooling with
whole-graph mean pooling and the same engineered features. Region marking lets
the graph representation depend on the proposed action. Permutation tests verify
the implemented behavior; they do not establish learned expressiveness or
generalization.

The frozen training specification uses eight independently seeded models per
family, AdamW with learning rate 0.001 and weight decay 0.0001, gradient clipping
at one, and thirty epochs. Each model retains the epoch with lowest supervised
validation loss. Both families share the same labeled data. Actual checkpoint
creation and inference profiling are pending at this evidence cutoff.

## Prospective experimental contract

| Role | Vertex counts | Replication |
| --- | --- | --- |
| Training supervision | 20, 24, 28, 32, 36, 40 | Sixteen states per size; eight model seeds per learned family |
| Validation supervision | 22, 26, 30, 34, 38 | Sixteen states per size |
| End to end validation | 22, 26, 30, 34, 38 | Eight model blocks, ten paired searches per size and block, four methods |
| Held out evaluation | 44, 48, 52, 56, 64, 72, 80, 96 | Same eight blocks, ten searches and four methods |

Controlled searches use the same verified CPU hardware and one experiment worker
at a time. Method order is balanced across blocks. The budget includes model
loading, initialization, candidate generation, features, inference and repair.
Checkpoint quality is recorded at one, ten and sixty seconds; sixty seconds is
primary. The complete validation panel contains 1,600 searches and 26.67 nominal
search-hours, excluding verification and orchestration. Held-out evaluation is
conditional on the validation gate and contains 2,560 searches and 42.67 nominal
search-hours. No held-out solver experiment has been run at this cutoff.

The declared learning gate requires a positive mean GNN advantage over adaptive
selection at sixty seconds in the complete validation panel. This is a
development gate, not the primary scientific success criterion. A failure stops
the larger neural campaign and leads to reproduction and a negative report.
After a pass, configurations, models, seeds, methods, hardware conditions and
analysis must be frozen before held-out outcomes are observed.

For each classical comparator, ten paired outcomes per size are averaged within
each model-seed block, then averaged over the declared sizes. The eight block
differences are the units of the paired analysis. Student t intervals with
Bonferroni adjustment across the two designated GNN comparisons give simultaneous
95% intervals, conditional on independent blocks and approximately normal block
means. Individual graph samples are not independent trained-model replicates.
The held-out primary target requires a mean advantage of at least one edge
against each classical comparator, both interval lower bounds above zero, and
positive mean advantages against both on at least six of eight sizes. MLP
comparisons and earlier checkpoints remain secondary.

## Completed development evidence

The first feasibility recipe improved five of 100 states and failed its gate.
That result remains retained. The revised recipe, chosen during development,
improved 28 of 100 states: sixteen at size 35 and twelve at size 40. None of these
repairs exceeded the strongest observed starting graph at its size. Passing this
gate permits supervised collection; it establishes recovery of weaker states,
not a learned advantage or demonstrated escape from the strongest construction.

Classical tuning completed 96 sixty-second searches on sizes 25, 31, 35 and 40,
using three paired starting seeds per trial. The predeclared selection rule used
mean sixty-second results, then ten seconds, one second and trial identifier to
resolve ties. The selected tabu trial averaged 96.000 edges and the selected
adaptive trial averaged 94.583 across these development sizes. Separate
construction, simple-repair and random-selection calibration averaged 95.000,
94.917 and 94.583 edges. These are small development comparisons, not estimates
of unseen-size strength or trained-model replication.

The complete training dataset passed independent audit and input replay for all
96 states. It contains 3,445 repair outcomes, including 1,115 positive gains;
49 states have differing region gains. The audit checked 5,950 graph occurrences,
including repeated witnesses, and retained 122 equal-score replacements without
a strict improvement event. Its 1,432 checks of local optimal status flags do not
constitute independently checked optimality proofs.

| State source | States | States with differing region gains | Positive repair outcomes |
| --- | ---: | ---: | ---: |
| Best generated start | 24 | 4 | 26 |
| Trimmed construction | 18 | 0 | 0 |
| Extended construction | 18 | 9 | 109 |
| Uniform greedy construction | 18 | 18 | 616 |
| Perturbed best start | 18 | 18 | 364 |
| Total | 96 | 49 | 1,115 |

Best repaired edge counts match the strongest initial counts at all training
sizes: 46, 59, 76, 92, 106 and 125 respectively. The first four equal published
exact values at sizes 20, 24, 28 and 32, so those strongest starts cannot admit
an additional edge. Published exact values at sizes 36 and 40 are 110 and 127.
This exposes a ceiling in much of the strongest-start training subset and helps
explain why useful observed signal is concentrated in weaker states. The table
is a reference for these development values, not an exhaustive certification of
current larger-size records. [Afzaly and McKay extremal graph data](https://users.cecs.anu.edu.au/~bdm/data/extremal.html).

## Reproducibility and costs

The retained source snapshot passed 179 software tests. Repair tests compare
every feasible order-four graph and every nonempty region with an independent
exhaustive oracle, and check an order-five global repair and timeout. Graph
updates and rollback are exhaustively checked at order five; other tests cover
deterministic work-limited replay and permutation behavior. These tests and earlier untrained
timing fixtures establish software behavior, not research performance.
Independent programs validate stored witnesses, provenance, seed panels and
recorded timing consistency. They do not externally remeasure historical timing.
Input replay reproduces generated inputs, rather than claiming bit-identical
wall-clock search trajectories on another machine.

The prospective timed reproduction subset is block zero, starting replicate
zero, all four methods at sizes 22, 30 and 38 for validation, or 44, 64 and 96
for held-out evaluation. It requires unchanged source and checkpoints in a
distinct locked software environment. Repetition on the same physical CPU is
software-environment reproduction and must not be described as independent
hardware replication. The final timed reproduction has not yet been performed.

Paid compute and storage spending is zero at this cutoff, within the EUR 500
campaign ceiling. The completed training-label collection used 1.321 local
worker-hours. Local computation is an incurred resource cost even when no paid
provider is used. The sequential campaign books completed stages once and will
report labeling, tuning and training costs separately. Matched profiling of
actual checkpoints and an inference-only scenario of thirty selections per
search will estimate conditional cost recovery. Nonpositive latency savings
give no finite inference-only recovery count; this scenario cannot replace
end-to-end solution-quality comparisons.

## Interpretation and outstanding work

The established results are a tested repair system, a passed revised feasibility
gate, stronger classical development comparisons and verified supervision.
Transferable learned advantage, superiority of the GNN over the MLP, a successful
held-out primary comparison and a new extremal lower bound remain unestablished.

Important limitations are the small supervision panel, many equal-gain states,
optimal starts at four training sizes, labels measured under short stochastic
repair budgets, and the risk that a selector learns to repair weak starts without
escaping stronger plateaus. A single CPU worker controls experiment concurrency
but cannot eliminate every background operating-system workload. Eight trained
replicates provide limited precision, and statistical interval assumptions must
be stated with the eventual outcomes.

The immediate work is complete validation-label collection and verification,
actual training and matched inference profiling, followed by the full end-to-end
validation gate. Subsequent held-out evaluation and exploratory record attempts
follow their declared conditions. A finished negative study can answer the
research question within this protocol. Publication strength will depend on the
result and its explanatory value; completion of software alone is insufficient.

## Evidence references

The study directory retains the [progress log](PROGRESS.md),
[training-label diagnostics](labels_train_v1/LABEL_QC.json),
[training witness audit](labels_train_v1/INDEPENDENT_AUDIT.json),
[training input replay](labels_train_v1/INPUT_REPLAY.json),
[revised feasibility evidence](feasibility_v2/DESCRIPTIVE_STATE.json),
[classical tuning selection](tuning_v1/SELECTION.json),
[calibration summary](development_calibration_v1/summary.json), and
[software-check record](SOFTWARE_CHECKS_179.json).
[MANUSCRIPT_EVIDENCE_v1.json](MANUSCRIPT_EVIDENCE_v1.json) binds this draft to the
completed evidence and reference checks used on 2 October 2026. Its cutoff
precedes actual trained checkpoints and controlled validation outcomes; future
results must be incorporated with new evidence bindings.

# Formulation audit during feasibility v2

This audit was recorded during the live v2 run; that run has now completed.
Its source/protocol remain frozen. The formulation findings and the original
gap list below describe the source archived with commit `177f974`. Later
implementation progress is recorded separately at the end of this document.

## What the formulation establishes

A simple graph is C4-free iff every pair of distinct vertices has at most one
common neighbor. This includes adjacent pairs: the objective forbids C4 as a
subgraph, not just an induced C4. Triangles remain allowed.

For a selected region R, every edge touching R is a Boolean variable; the complement
is fixed. The original C4-free graph satisfies the model and remains an incumbent.
Every lazily added cut has form `sum(variable cycle edges) <= 3 - fixed cycle edges`.
It forbids an actual four-cycle without forbidding a feasible graph.

An OPTIMAL solution of a relaxation that is independently C4-free certifies a maximum
for that fixed complement: its value is both a feasible lower bound and a relaxation
upper bound. This does not establish the global graph optimum unless R is all vertices.
A feasible timeout does not establish a local optimum.

## Incorrect comment about initial constraints

The engine says its initial strengthening excludes every C4 with at most two selected
vertices. It excludes the one-selected-vertex case and the case of two **opposite**
selected vertices. It does **not** exclude all cases with two adjacent selected vertices.

Counterexample: vertices 0,1,2,3; R={0,1}; fixed complement edge {2,3}; proposed
cycle edges {0,1}, {1,2}, {2,3}, {0,3}. The initial constraints all hold:

- Outside vertices 2 and 3 have no common neighbor in the fixed complement, so no
  one-region-vertex cut applies to that pair.
- For a selected vertex and an outside vertex there is at most one common neighbor
  through the complement.
- Selected vertices 0 and 1 have no common neighbor through the complement.

Nevertheless the proposal has the C4 0–1–2–3–0. The separation step adds
`x01 + x12 + x03 <= 2`, rejects it, and keeps the feasible incumbent. The final
feasibility guarantee is intact; the initial-model comment overstates its strength.

After v2 finishes, correct the comment and investigate adding all three-variable
cuts for two adjacent selected vertices. For every selected pair r,s and fixed edge
a,b, the two necessary inequalities are:

```
x_rs + x_ra + x_sb <= 2
x_rs + x_rb + x_sa <= 2
```

Their construction/search overhead must be charged. Benchmark the revision on
development data before claiming it is faster; adding constraints can also slow CP-SAT.
Do not retroactively describe v2 as using them.

## Further work before the controlled comparison

1. Require a complete independent audit before freezing, including all four methods,
   witness feasibility, expected seeds, checkpoint identities, timing events, identical
   starting pools, source hashes and every failed record. Currently `freeze()` relies
   on its analysis input rather than calling a full evidence verifier.
2. Show MLP outcomes alongside the primary GNN–tabu and GNN–adaptive comparisons.
   The MLP is a necessary learned-selection control; its result cannot replace a
   failed primary GNN outcome.
3. Add a timed version of the construction/simple-repair calibration controls.
   The frozen original pilot remains available but is not a matched-time comparison.
4. Tune classical parameters using development data and retain that tuning evidence.
   Two-second smoke runs do not establish that the current baselines are strong.
5. Randomize or balance method order within each paired search group before the
   final freeze. The current fixed order can confound methods with time-dependent
   machine load, temperature, and cache behavior.
6. Enforce the experimental concurrency policy in the runner; recording one CP-SAT
   worker is not by itself a process-level exclusion of simultaneous experiments.
7. Produce a failed/incomplete-panel report rather than only raising an exception.
   Retained errors must be visible in the scientific account.
8. Record the actual CPU model before any primary freeze. Current sandbox metadata
   can identify only arm64 and ten logical CPUs; it explicitly says the model is
   unverified. Resolve this before, not after, the controlled comparison.
9. Replay declared starting-pool and candidate-generation inputs from the source
   snapshot. Set-based witness checks verify graph feasibility but do not by themselves
   prove that a generated pool or label panel is complete.
10. Let adaptive selection learn **family × region-size** preferences, rather than
    family alone. The current selector samples sizes uniformly within its chosen
    family, even when three-vertex repairs are much cheaper than larger repairs.
    Otherwise a learned selector could win merely by choosing the cheap size, while
    the designated adaptive comparator is prevented from adapting to it. Retain
    exploration, measure end-to-end development performance, and apply the same
    candidate generation, repair, diversification and acceptance code to all selectors.

These are unfinished campaign requirements. None authorizes changing a live frozen
gate, running held-out sizes before freezing, or claiming a learned search advantage.

## Implementation progress after v2 completion

The revised gate passed 28/100, but its best repaired graphs did not exceed the
strongest initial construction at either size. No learned advantage is established.

The current runner integrates full independent audit and input-replay requirements,
MLP reporting, timed construction/simple-repair calibration, paired method-order
balancing, process-level exclusion, descriptive failure reporting, and adaptive
family/size selection. Architecture and training-source compatibility are checked
when loading research checkpoint panels. A development-only classical tuning
runner now retains all trials and its predeclared selection rule.

These are implementation changes with 151 passing software tests. Classical tuning,
training, end-to-end validation and the final held-out freeze remain unfinished.
Hardware identification is available through a permitted CPU-model read; future
timed contracts must actually retain a verified model and preserve that environment
on resume. Existing unverified frozen metadata is not retroactively altered.

The initial-strengthening comment and equal-score repair-event limitation above
remain disclosed errata for the unchanged repair machinery. No adjacent-pair cuts
or new tie-acceptance behavior have been introduced into the passed v2 recipe.

## Equal-score incumbent witnesses

The frozen engine's `accept()` replaces its incumbent when a feasible proposal has
**at least** the current edge count and has been checked before the deadline. Its
event list stores only **strict improvements**. Consequently, the final repair graph
can differ from the last timed improvement graph while having the same edge count.
This also occurs on zero-gain repairs relative to the initial graph.

The independent audit verifies both witnesses and their fixed complements, and
checks that the reported gain has an explicit within-budget improvement witness.
It reports equal-score replacements lacking their own event separately; it does not
assign the earlier witness's timestamp to the replacement. This is a provenance
limitation, not an additional gain or evidence of an infeasible output. Timed search
results must still match their credited final history witness exactly.

For subsequent repair versions, retain the final accepted-witness time (including
ties), or return the explicitly timed witness when describing a timed result. Keep
the existing v2 evidence unchanged. Any resulting execution change must be profiled
and independently gated before the learned campaign uses it.

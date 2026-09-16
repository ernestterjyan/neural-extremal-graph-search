# Mathematical and diagnostic contract

For a simple undirected graph G on n≥2 vertices, an absent edge uv can be added without creating a triangle exactly when N(u)∩N(v) is empty. The environment supplies this legal action set. Since edges are never removed, every episode terminates at a maximal triangle-free graph. Maximal means no legal addition remains; maximum means the edge count reaches T(n)=floor(n²/4).

Mantel's theorem identifies T(n) and its balanced complete bipartite equality case. The policy maximizes expected terminal |E|/T(n). The cross-entropy update imitates actions from high-return sampled trajectories; it is not a proof procedure. Reporting both density and exact success is essential: losing one edge and losing the possibility of any optimal completion answer different questions.

## Exact completion feasibility

For each connected component of a bipartite partial graph, let its color-class sizes be (a_i,b_i). A balanced bipartite completion exists if and only if one can independently choose either a_i or b_i from each component so that their sum is floor(n/2). The other side then has ceil(n/2) vertices. Isolated vertices contribute (1,0).

Necessity follows because a connected component's bipartition is unique up to reversal. Sufficiency follows by orienting the components according to such a choice and adding all missing edges across the resulting global partition. A bitset subset-sum computation checks the condition exactly, including odd n. An odd cycle immediately makes bipartite completion impossible. Since edges cannot be removed, loss of balanced-completion feasibility is irreversible.

After a proposed edge, components either merge with a forced relative orientation, stay bipartite in the same component, or acquire an odd cycle. The implementation caches the component-merger cases when annotating all candidate actions. Small cases are checked against independent enumeration of every balanced vertex partition.

Every terminal bipartite output is complete bipartite: an absent cross-partition edge could otherwise be added, and disconnected components would also admit a legal connecting edge. Therefore terminal outcomes split into exact balanced optimum, unbalanced complete bipartite, and nonbipartite. The first irreversible loss can be partition imbalance even when an odd cycle appears later. These two events are recorded separately.

## Symmetry and action aliasing

On balanced K(m,m), every vertex has the same initial supplied features. Shared equivariant mean aggregation preserves equality of their embeddings at every layer. Identical terminal embeddings are therefore required by this architecture; terminal clustering cannot diagnose a learned partition.

On C8, all vertices have degree 2 and three incident legal actions. Every candidate has identical supplied statistics. Mean message passing preserves equal node embeddings at any depth, and the symmetric endpoint scorer consequently gives all 12 candidates the same score. Eight distance-three choices admit 16-edge completions. Four opposite-vertex chords admit at most 13 edges. Exhaustively checking all 2^12 subsets of initially legal edges establishes those continuation values, with independent triangle checks and persisted maximizing witnesses.

Hence the bad-chord probability at this particular state is 1/3 and the expected final edge count conditioned on taking an action there is at most (8×16+4×13)/12=15, even with optimal continuation afterward. This is a 15/16 conditional ratio bound. It places no corresponding unconditional ceiling on rollouts from the empty graph, which may avoid C8.

The diagnostic collection also includes C6, C7, two disjoint C4s, C4 with two isolates, and a three-edge matching. These examples distinguish regularity, initial infeasibility, and action classes with equal continuation values. Every reported continuation value comes from enumerating all initially legal subsets, a monotone superset of every possible continuation, rejecting any subset containing a triangle. Witnesses are verified separately with NetworkX.

For real trajectories, symbolic node colors track initial degree/legal-incidence features and exact normalized neighbor-color multiplicities through the encoder's specified depth. Candidate signatures combine endpoint colors and supplied edge statistics. Equal signatures are a sufficient condition for identical representations in exact arithmetic for all weights, not a complete characterization of fitted-logit equality. A mixed group contains both completion-preserving and completion-destroying actions. Its presence alone does not force an error: a distinguishable safe group may still be chosen. If every group has a bad-action fraction greater than zero, the smallest such fraction lower-bounds bad-action probability for this representation at that state. Finite-precision roundoff can perturb exact equalities slightly; it is not evidence of robust access to the missing relation.

These annotations are not primary policy inputs or training rewards. Observing a chosen bad action inside a mixed group establishes consequential aliasing locally; it does not establish that all failures, or a whole trajectory's counterfactual performance, are caused by that limitation.

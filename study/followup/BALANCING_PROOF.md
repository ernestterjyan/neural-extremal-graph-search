# Why the component-balancing control succeeds

This is an elementary property of the explicitly constrained policy, not a new extremal-graph theorem. The empirical result is 2,000 optimal graphs over the declared four sizes and five seed blocks; the argument below covers every tie choice for every n≥2 under the specified rules.

The environment permits only triangle-free edge additions. The parity mask further permits an edge within a connected bipartite component only when it joins opposite color classes. It permits all edges between different components. For every permitted edge, the balancing rule computes the absolute difference between the color-class sizes of the connected component containing that edge **after** adding it. It selects the smallest such difference, breaking ties uniformly.

Initially every vertex is an isolated component. An isolated vertex has class sizes (1,0). Suppose that, so far, the policy has formed only isolated vertices and nontrivial components with equal color-class sizes.

1. Any edge between two isolates creates a balanced (1,1) component, so its score is zero.
2. Any edge between two balanced components creates another balanced component, whichever relative orientation the endpoints force, so its score is zero.
3. If a balanced component is missing a cross-class edge internally, that edge is legal and parity-permitted, and its score is zero.
4. Joining an isolate to a balanced component creates a component with class sizes differing by one, so its score is one.

Therefore, while a score-zero move exists, every selected move preserves the stated form. Once no score-zero move exists, there can be at most one isolate, at most one nontrivial balanced component, and that component must be complete bipartite. If n is even, the isolate cannot be present by parity of the vertex count; the graph is already the optimum K(n/2,n/2). If n is odd, exactly one isolate remains. Attaching it gives a single connected bipartite component with part sizes (floor(n/2),ceil(n/2)). Any subsequent permitted edge is a missing cross-class edge in this component. Adding such edges preserves its part sizes until the complete balanced bipartite graph is reached.

This proves optimal termination for every n≥2 and every tie choice. The rule uses exact component coloring and class sizes supplied by the algorithm, information the frozen neural scorers do not explicitly receive as a post-processing decision rule. Its success demonstrates that the known optimum has a simple local constructive policy once component parity is made available. It does not prove anything about the internal computation of the GNN. The implementation was additionally checked against independent NetworkX colorings on small states and all tied choices exhaustively through n=6.

# Representative trajectories

Each JSONL file contains a compact sample of elite trajectories retained by the
corresponding training seed. A record includes `n`, `r`, optimality ratio, the
ordered edge actions, and the final serialized graph.

The `mvp-seed-*` samples come from the GNN and `mvp-mlp-seed-*` from the
fixed-size control. These samples are illustrative. Quantitative results come
from the complete episode-level files under `results/mvp/` and
`results/mvp-v0.2/`.

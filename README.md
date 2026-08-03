# Neural Extremal Graph Search

Can a graph neural network trained only on small graphs learn the construction
behind Turán's theorem and transfer it to larger, unseen graph sizes?

**MVP status: complete.** The trained policy reached 91.82% mean optimality and
29.5% exact-optimum success at the unseen size \(n=24\), versus 62.76% and 0%
for uniform random construction. Across 40,000 evaluated graphs there were no
constraint violations and every terminal graph was maximal.

This repository is a controlled mathematical machine-learning experiment. An
agent starts with an empty graph, repeatedly adds an edge without creating a
triangle, and is rewarded for ending with as many edges as possible. The exact
optimum is known: the balanced complete bipartite Turán graph \(T_2(n)\).

## MVP capabilities

- exact Turán bounds and graph construction for general \(r\);
- a bitset-backed triangle-free graph-building environment;
- independent constraint and maximality verification;
- uniform-random and least-degree baselines;
- a custom permutation-equivariant PyTorch edge policy;
- elite-trajectory cross-entropy training with a size curriculum;
- frozen-policy evaluation on both training and unseen graph sizes;
- seed-level result aggregation and publication-ready figures.

The MVP deliberately excludes \(K_4\)-free construction, fixed-size MLPs,
ablation studies, embedding analysis, and an interactive application.

## Main result

| Size | Split | GNN optimality | Exact optimum | Random optimality |
|---:|:---|---:|---:|---:|
| 14 | trained | 98.20% | 84.5% | 74.50% |
| 16 | unseen | 97.45% | 76.1% | 71.71% |
| 20 | unseen | 95.08% | 54.3% | 66.95% |
| 24 | unseen | 91.82% | 29.5% | 62.76% |

The evidence supports **strong but imperfect size generalization**. The policy
learned a transferable construction heuristic, but performance and exact
success declined as evaluation moved farther beyond the training range.

Read the [final report](reports/final_report.md), inspect the
[complete summary](results/mvp/summary.md), or view the
[optimality plot](reports/figures/optimality_ratio.png).

## Quick start

Install [uv](https://docs.astral.sh/uv/), then create the isolated environment:

```bash
uv sync --extra dev
```

Run the fast smoke experiment first:

```bash
uv run negs train --config experiments/configs/smoke.toml --seed 0
uv run negs evaluate --config experiments/configs/smoke-eval.toml
```

Run the five full curriculum seeds (each command is independently resumable):

```bash
uv run negs train --config experiments/configs/mvp.toml --seed 0
uv run negs train --config experiments/configs/mvp.toml --seed 1
uv run negs train --config experiments/configs/mvp.toml --seed 2
uv run negs train --config experiments/configs/mvp.toml --seed 3
uv run negs train --config experiments/configs/mvp.toml --seed 4
```

After all five seeds finish:

```bash
uv run negs evaluate --config experiments/configs/mvp-eval.toml
uv run negs report --results results/mvp/evaluation.csv
```

Verify a serialized graph independently:

```bash
uv run negs verify path/to/graph.json
```

See [configuration](docs/configuration.md), [public API](docs/api.md), and the
[final report](reports/final_report.md) for details.

## Recreate the reported figures

The curated episode-level evaluation data and all five trained checkpoints are
included. Regenerate the summary tables and figures with:

```bash
uv run negs report --results results/mvp/evaluation.csv
```

The release checkpoints are under `artifacts/checkpoints/`; compact example
trajectories are under `artifacts/trajectories/`.

## Reproducibility

Canonical reported runs use CPU and seeds `0, 1, 2, 3, 4`. Every run records
its resolved configuration, Python and dependency versions, platform, seed,
metrics, sampled trajectories, and best checkpoint. MPS can be selected for
exploratory runs, but bit-for-bit reproducibility is only required on CPU.

The primary metric is

\[
\frac{|E(G_T)|}{|E(T_2(n))|}.
\]

Reported uncertainty intervals are 95% Student-t intervals over seed-level
means. The full evaluation uses 200 episodes for every method, size, and seed.

## Related work and positioning

This project is inspired by Wagner-style reinforcement learning for graph
theory, including:

- [A Systematization of the Wagner Framework](https://arxiv.org/abs/2406.12667);
- [the CuriosAI reference implementation](https://github.com/CuriosAI/graph_conjectures);
- [RLGT](https://arxiv.org/abs/2602.17276).

The contribution here is narrower: an exactly verifiable study of size
extrapolation and structural construction against a known extremal theorem.

## License

MIT

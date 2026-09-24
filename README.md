# Neural Extremal Graph Search

**Corrected study (September 2026):** twenty new training runs and 97,000 verified research graphs. See the [research assessment](study/RESEARCH_REPORT.md) and [reproduction guide](study/README.md). The subsequent [parity mechanism follow-up](study/followup/README.md) tests the main unresolved mechanism against stronger controls. The v0.1/v0.2 discussion below is preserved historical context; its GNN–MLP comparison used different training versions.

[![CI](https://github.com/ernestterjyan/neural-extremal-graph-search/actions/workflows/ci.yml/badge.svg)](https://github.com/ernestterjyan/neural-extremal-graph-search/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ernestterjyan/neural-extremal-graph-search)](https://github.com/ernestterjyan/neural-extremal-graph-search/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Can a graph neural network trained only on small graphs learn the construction
behind Turán's theorem and transfer it to larger, unseen graph sizes?

**v0.2 study status: complete.** The trained GNN reached 91.82% mean
optimality and 29.5% exact-optimum success at the unseen size \(n=24\), versus
62.76% and 0% for uniform random construction. Across 40,000 evaluated graphs
there were no constraint violations and every terminal graph was maximal.

The fixed-size MLP control reached 91.55% mean optimality but only 17.3% exact
success at \(n=24\). Across the combined 56,000-graph evaluation there were
again zero violations and every terminal graph was maximal.

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
- a parameter-matched, position-sensitive fixed-size MLP control;
- elite-trajectory cross-entropy training with a size curriculum;
- frozen-policy evaluation on both training and unseen graph sizes;
- seed-level result aggregation and publication-ready figures.

The project still excludes \(K_4\)-free construction, embedding analysis, and
an interactive application.

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

## v0.2 fixed-size control

The MLP has 37,990 trainable parameters versus 38,337 for the GNN and reuses
the same environment, features, legal-action mask, training algorithm, seeds,
and evaluation protocol. Unlike the GNN, it depends on absolute padded vertex
positions and is deliberately not permutation equivariant.

| Size | Split | GNN ratio | MLP ratio | GNN exact | MLP exact |
|---:|:---|---:|---:|---:|---:|
| 14 | trained | 98.20% | 97.12% | 84.5% | 53.4% |
| 16 | unseen | 97.45% | 95.50% | 76.1% | 44.2% |
| 20 | unseen | 95.08% | 93.99% | 54.3% | 31.8% |
| 24 | unseen | 91.82% | 91.55% | 29.5% | 17.3% |

The control weakens a simple architecture-only explanation: a non-equivariant
MLP also learned a strong transferable heuristic. The GNN nevertheless had
higher exact-optimum success at every tested size and better mean optimality
through most of the range. Read the
[v0.2 research addendum](reports/v0.2_report.md) and
[complete v0.2 summary](results/mvp-v0.2/summary.md).

To understand or extend the implementation, start with the
[architecture guide](docs/architecture.md).

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

Exercise the fixed-size MLP control end to end:

```bash
uv run negs train --config experiments/configs/mlp-smoke.toml --seed 0
uv run negs evaluate --config experiments/configs/mlp-smoke-eval.toml
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

The corresponding fixed-size control uses `mvp-mlp.toml`; the combined frozen
evaluation uses `mvp-v0.2-eval.toml`.

Verify a serialized graph independently:

```bash
uv run negs verify path/to/graph.json
```

See the [architecture guide](docs/architecture.md),
[configuration reference](docs/configuration.md), [public API](docs/api.md),
and [final report](reports/final_report.md) for details.

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

## Development

Read the [architecture guide](docs/architecture.md) before changing model,
feature, checkpoint, or evaluation contracts. The [changelog](CHANGELOG.md)
records release-level changes. Run the complete local quality gate with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Related work and positioning

This project is inspired by Wagner-style reinforcement learning for graph
theory, including:

- [A Systematization of the Wagner Framework](https://arxiv.org/abs/2406.12667);
- [the CuriosAI reference implementation](https://github.com/CuriosAI/graph_conjectures);
- [RLGT](https://arxiv.org/abs/2602.17276).

The contribution here is narrower: an exactly verifiable study of size
extrapolation and structural construction against a known extremal theorem.

## Citation

If you use this software or its results, cite the metadata in
[CITATION.cff](CITATION.cff) and the archived
[v0.2.0 release](https://github.com/ernestterjyan/neural-extremal-graph-search/releases/tag/v0.2.0).

## License

MIT

## Corrected research study

A separate, prospectively specified study repairs sampling and resume provenance, retrains GNN/MLP and simpler controls under one implementation, and retains every evaluated graph. See [the study guide](study/README.md), [frozen protocol](study/protocol.json), and [research report](study/RESEARCH_REPORT.md). Historical release artifacts above remain unchanged and should not be pooled with this study.

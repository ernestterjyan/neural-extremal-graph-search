"""Independent diagnostic checks and batching replay of selected persisted study graphs."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.rollouts import run_episode_batch  # noqa: E402
from extremal_graph.serialization import graph_to_dict  # noqa: E402
from extremal_graph.study import digest  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.utils import write_json  # noqa: E402


def independent_prefix_diagnostics(n, actions):
    g = nx.empty_graph(n)
    first_loss = None
    first_odd = None
    cause = None
    for step, edge in enumerate(actions, 1):
        g.add_edge(*edge)
        try:
            colors = nx.bipartite.color(g)
        except nx.NetworkXError:
            first_odd = step
            if first_loss is None:
                first_loss, cause = step, "odd_cycle"
            break
        reachable = {0}
        for component in nx.connected_components(g):
            a = sum(colors[v] for v in component)
            b = len(component) - a
            reachable = {total + side for total in reachable for side in [a, b]}
        if n // 2 not in reachable and first_loss is None:
            first_loss, cause = step, "partition_imbalance"
    if not nx.is_bipartite(nx.Graph(actions)):
        outcome = "nonbipartite"
    elif len(actions) == n * n // 4:
        outcome = "exact"
    else:
        outcome = "unbalanced_bipartite"
    return dict(
        first_loss_step=first_loss,
        first_loss_cause=cause,
        first_odd_cycle_step=first_odd,
        outcome=outcome,
    )


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    diagnostics = 0
    replays = 0
    for family in [
        "gnn",
        "mlp",
        "candidate",
        "endpoint",
        "random",
        "least_degree",
        "lookahead",
        "untrained_gnn",
        "untrained_mlp",
    ]:
        for seed in range(5):
            model = None
            if family in ["gnn", "mlp", "candidate", "endpoint"]:
                model, _ = load_model_checkpoint(
                    ROOT / f"study/artifacts/training/corrected-{family}-seed-{seed}/best.pt"
                )
            for n in [14, 24, 40]:
                if "mlp" in family and n > 24:
                    continue
                cell = ROOT / f"study/artifacts/evaluation/{family}/seed-{seed}/n-{n}"
                with gzip.open(cell / "chunk-000000.json.gz", "rt") as handle:
                    records = json.load(handle)["records"][:2]
                for record in records:
                    value = independent_prefix_diagnostics(n, record["actions"])
                    if any(record["row"][k] != v for k, v in value.items()):
                        raise ValueError(
                            f"independent diagnostic mismatch: {record['row']['record_id']}"
                        )
                    diagnostics += 1
                if model is not None:
                    seeds = [r["row"]["episode_seed"] for r in records]
                    grouped = run_episode_batch(model, n=n, seeds=list(reversed(seeds)))
                    alone = run_episode_batch(model, n=n, seeds=[seeds[0]])[0]
                    expected = {r["row"]["episode_seed"]: r["row"]["graph_sha256"] for r in records}
                    for s, t in zip(reversed(seeds), grouped, strict=True):
                        if digest(graph_to_dict(t.final_state)) != expected[s]:
                            raise ValueError("saved-checkpoint batching replay mismatch")
                        replays += 1
                    if digest(graph_to_dict(alone.final_state)) != expected[seeds[0]]:
                        raise ValueError("single-episode replay mismatch")
                    replays += 1
    result = {
        "independent_trajectory_diagnostics": diagnostics,
        "checkpoint_replays": replays,
        "valid": True,
        "selection": (
            "episodes 0 and 1, all seeds/methods at n14/24/40 (supported cells); "
            "learned families also single/reversed-pair replay"
        ),
    }
    write_json(ROOT / "study/independent_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()

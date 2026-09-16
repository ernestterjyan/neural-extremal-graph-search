"""Exploratory, prospectively specified parity intervention with information-matched control."""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.diagnostics import components, trajectory_diagnostics  # noqa: E402
from extremal_graph.env import GraphConstructionEnv  # noqa: E402
from extremal_graph.features import collate_graph_states  # noqa: E402
from extremal_graph.graph import GraphState, Trajectory  # noqa: E402
from extremal_graph.policy import sample_actions  # noqa: E402
from extremal_graph.serialization import graph_to_dict  # noqa: E402
from extremal_graph.study import collect, digest, independent_verify, verify_bundle  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.utils import (  # noqa: E402
    append_jsonl,
    environment_metadata,
    sha256_file,
    write_json,
)


def parity_keep(state, legal):
    info = components(state)
    if not info.bipartite:
        raise ValueError("parity-protected trajectory became nonbipartite")
    return [
        info.component[u] != info.component[v] or info.colors[u] != info.colors[v] for u, v in legal
    ]


def generate(model, method, n, seeds):
    envs = [GraphConstructionEnv(n) for _ in seeds]
    generators = [torch.Generator().manual_seed(s) for s in seeds]
    states = [[] for _ in seeds]
    actions = [[] for _ in seeds]
    rewards = [[] for _ in seeds]
    active = list(range(len(envs)))
    while active:
        current = [envs[i].state for i in active]
        legal = [envs[i].legal_edges() for i in active]
        batch = collate_graph_states(current, legal)
        with torch.inference_mode():
            logits = (
                model(batch)
                if model is not None
                else torch.zeros_like(batch.candidate_features[:, :, 0]).masked_fill(
                    ~batch.candidate_mask, -torch.inf
                )
            )
            mask = batch.candidate_mask.clone()
            if method != "gnn_fresh":
                for j, (state, edges) in enumerate(zip(current, legal, strict=True)):
                    mask[j, : len(edges)] = torch.tensor(parity_keep(state, edges))
                if not mask.any(dim=1).all():
                    raise ValueError("parity filter cannot terminate a nonterminal bipartite graph")
                logits = logits.masked_fill(~mask, -torch.inf)
            indices = sample_actions(
                logits, candidate_mask=mask, generators=[generators[i] for i in active]
            )
        next_active = []
        for j, i in enumerate(active):
            index = int(indices[j])
            if not 0 <= index < len(legal[j]):
                raise ValueError("invalid intervention action")
            edge = legal[j][index]
            states[i].append(current[j])
            actions[i].append(edge)
            transition = envs[i].step(edge)
            rewards[i].append(transition.reward)
            if not transition.terminated:
                next_active.append(i)
        active = next_active
    return [
        Trajectory(
            n, 2, tuple(s), tuple(a), tuple(r), env.state, env.state.edge_count / (n * n // 4)
        )
        for s, a, r, env in zip(states, actions, rewards, envs, strict=True)
    ]


def self_check():
    from extremal_graph.baselines import UniformRandomPolicy, run_baseline_episode

    torch.set_num_threads(1)
    for n in range(4, 9):
        trajectory = run_baseline_episode(n, UniformRandomPolicy(), seed=99 + n)
        for state in trajectory.states:
            if not components(state).bipartite:
                break
            env = GraphConstructionEnv(n)
            for edge in state.edges:
                env.step(edge)
            legal = env.legal_edges()
            actual = parity_keep(state, legal)
            expected = []
            for edge in legal:
                graph = nx.empty_graph(n)
                graph.add_edges_from(state.edges + (edge,))
                expected.append(nx.is_bipartite(graph))
            assert actual == expected
    c8 = GraphState(8, 2, tuple((i, (i + 1) % 8) for i in range(8)))
    env = GraphConstructionEnv(8)
    for edge in c8.edges:
        env.step(edge)
    legal = env.legal_edges()
    keep = parity_keep(c8, legal)
    assert sum(keep) == 8
    assert all((v - u) % 2 == 1 for (u, v), yes in zip(legal, keep, strict=True) if yes)
    together = generate(None, "uniform_parity", 8, [7, 9])
    assert together[0].actions == generate(None, "uniform_parity", 8, [7])[0].actions
    from extremal_graph.rollouts import run_episode_batch

    model, _ = load_model_checkpoint(ROOT / "artifacts/checkpoints/mvp-seed-0.pt")
    fresh = generate(model, "gnn_fresh", 8, [7, 9])
    canonical = run_episode_batch(model, n=8, seeds=[7, 9])
    assert [x.actions for x in fresh] == [x.actions for x in canonical]
    trained, _ = load_model_checkpoint(
        ROOT / "study/artifacts/training/corrected-gnn-seed-0/best.pt"
    )
    larger = generate(trained, "gnn_fresh", 40, [9_000_100, 9_000_101])
    expected = run_episode_batch(trained, n=40, seeds=[9_000_100, 9_000_101])
    assert [x.actions for x in larger] == [x.actions for x in expected]
    return {
        "networkx_mask_check": True,
        "C8_retained": 8,
        "batch_consistency": True,
        "unmodified_arm_matches_canonical": True,
    }


def cell(job):
    method, seed, n = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / "study/intervention-protocol.json").read_text())
    checkpoint = ROOT / f"study/artifacts/training/corrected-gnn-seed-{seed}/best.pt"
    model = None
    if method != "uniform_parity":
        model, _ = load_model_checkpoint(checkpoint)
    output = ROOT / f"study/artifacts/intervention/{method}/seed-{seed}/n-{n}"
    contract = {
        "method": method,
        "seed": seed,
        "n": n,
        "episodes": protocol["episodes_per_cell"],
        "batch_size": protocol["batch_size"],
        "protocol_sha256": sha256_file(ROOT / "study/intervention-protocol.json"),
        "script_sha256": sha256_file(__file__),
        "checkpoint_sha256": sha256_file(checkpoint) if model is not None else None,
    }
    fingerprint = digest(contract)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["contract_hash"] != fingerprint:
            raise ValueError("intervention contract mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest_path,
            {
                "status": "running",
                "contract": contract,
                "contract_hash": fingerprint,
                "environment": environment_metadata(torch.device("cpu")),
            },
        )
    chunks = {}
    for start in range(0, protocol["episodes_per_cell"], protocol["batch_size"]):
        path = output / f"chunk-{start:06}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as h:
                saved = json.load(h)
            if saved["contract_hash"] != fingerprint:
                raise ValueError("intervention chunk mismatch")
            chunks[path.name] = sha256_file(path)
            continue
        count = min(protocol["batch_size"], protocol["episodes_per_cell"] - start)
        seeds = [(101 + seed) * 1000000 + n * 10000 + e for e in range(start, start + count)]
        begun = time.perf_counter()
        try:
            trajectories = generate(model, method, n, seeds)
            seconds = time.perf_counter() - begun
            records = []
            for i, t in enumerate(trajectories):
                graph = graph_to_dict(t.final_state)
                if not all(independent_verify(t.final_state).values()):
                    raise ValueError("intervention graph invalid")
                row = {
                    "method": method,
                    "seed": seed,
                    "n": n,
                    "episode": start + i,
                    "episode_seed": seeds[i],
                    "record_id": f"{method}/s{seed}/n{n}/e{start + i}",
                    "edge_count": t.final_state.edge_count,
                    "optimal_edge_count": n * n // 4,
                    "optimality_ratio": t.optimality_ratio,
                    "exact_optimum": int(t.optimality_ratio == 1),
                    "absolute_gap": n * n // 4 - t.final_state.edge_count,
                    "inference_time_seconds": seconds / count,
                    "graph_sha256": digest(graph),
                    **trajectory_diagnostics(t),
                }
                records.append({"row": row, "graph": graph, "actions": t.actions, "probe": None})
            temporary = path.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt") as h:
                json.dump({"contract_hash": fingerprint, "records": records}, h)
            temporary.replace(path)
            chunks[path.name] = sha256_file(path)
        except Exception as error:
            write_json(
                output / "failure.json", {"error": repr(error), "status": "failed", "start": start}
            )
            raise
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="complete", chunks=chunks, rows=protocol["episodes_per_cell"])
    write_json(manifest_path, manifest)
    return str(output.relative_to(ROOT))


def main():
    protocol = json.loads((ROOT / "study/intervention-protocol.json").read_text())
    write_json(ROOT / "study/intervention-validation.json", self_check())
    jobs = [
        (m, s, n)
        for m in protocol["methods"]
        for s in protocol["training_seeds"]
        for n in protocol["sizes"]
    ]
    append_jsonl(
        ROOT / "study/ledger.jsonl",
        {
            "event": "intervention_launch",
            "time": time.time(),
            "jobs": len(jobs),
            "protocol_sha256": sha256_file(ROOT / "study/intervention-protocol.json"),
        },
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        for i, result in enumerate(pool.map(cell, jobs), 1):
            append_jsonl(
                ROOT / "study/ledger.jsonl",
                {"event": "intervention_completed", "cell": result, "time": time.time()},
            )
            print(f"intervention {i}/{len(jobs)}", flush=True)
    root = ROOT / "study/artifacts/intervention"
    result = verify_bundle(root)
    if result["cells"] != len(jobs) or result["graphs"] != 6000:
        raise ValueError("incomplete intervention matrix")
    write_json(ROOT / "study/intervention-verification.json", result)
    collect(root, root / "evaluation.csv")
    frame = pd.read_csv(root / "evaluation.csv")
    seed = frame.groupby(["method", "n", "seed"], as_index=False)[
        ["optimality_ratio", "exact_optimum", "absolute_gap", "inference_time_seconds"]
    ].mean()
    (ROOT / "study/tables").mkdir(exist_ok=True)
    seed.to_csv(ROOT / "study/tables/intervention_seed_results.csv", index=False)
    seed.groupby(["method", "n"], as_index=False).mean(numeric_only=True).to_csv(
        ROOT / "study/tables/intervention_summary.csv", index=False
    )
    print(result)


if __name__ == "__main__":
    main()

"""Frozen, information-controlled comparison of parity-protected policies."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parity_intervention import generate as generate_parity  # noqa: E402
from parity_intervention import parity_keep  # noqa: E402

from extremal_graph.diagnostics import components, trajectory_diagnostics  # noqa: E402
from extremal_graph.env import GraphConstructionEnv  # noqa: E402
from extremal_graph.graph import Edge, GraphState, Trajectory  # noqa: E402
from extremal_graph.policy import sample_actions  # noqa: E402
from extremal_graph.serialization import graph_from_dict, graph_to_dict  # noqa: E402
from extremal_graph.study import collect, digest, independent_verify, verify_bundle  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.utils import (  # noqa: E402
    append_jsonl,
    environment_metadata,
    sha256_file,
    source_manifest,
    write_json,
)

STUDY = ROOT / "study/followup"
ART = STUDY / "artifacts"
PROTOCOL = STUDY / "protocol.json"


def balance_scores(state: GraphState, legal: list[Edge]) -> list[int | None]:
    """Size imbalance of the edge's resulting component; None means forbidden."""
    info = components(state)
    if not info.bipartite:
        raise ValueError("balancing control requires a bipartite prefix")
    result = []
    for (u, v), allowed in zip(legal, parity_keep(state, legal), strict=True):
        if not allowed:
            result.append(None)
            continue
        i, j = info.component[u], info.component[v]
        a, b = info.sizes[i]
        if i == j:
            result.append(abs(a - b))
            continue
        c, d = info.sizes[j]
        flip = info.colors[u] ^ info.colors[v] ^ 1
        result.append(abs(a + c - b - d) if flip == 0 else abs(a + d - b - c))
    return result


def generate_balance(n: int, seeds: list[int]) -> list[Trajectory]:
    envs = [GraphConstructionEnv(n) for _ in seeds]
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    states = [[] for _ in seeds]
    actions = [[] for _ in seeds]
    rewards = [[] for _ in seeds]
    active = list(range(len(seeds)))
    while active:
        next_active = []
        for i in active:
            state = envs[i].state
            legal = envs[i].legal_edges()
            scores = balance_scores(state, legal)
            candidates = [v for v in scores if v is not None]
            if not candidates:
                raise ValueError("parity constraint left no action in a nonterminal graph")
            best = min(candidates)
            mask = torch.tensor([[score == best for score in scores]], dtype=torch.bool)
            logits = torch.zeros((1, len(legal))).masked_fill(~mask, -torch.inf)
            index = int(sample_actions(logits, candidate_mask=mask, generators=[generators[i]])[0])
            edge = legal[index]
            states[i].append(state)
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


def cell(job: tuple[str, int, int]) -> str:
    method, training_seed, n = job
    protocol = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(protocol["threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    family = method.removesuffix("_parity")
    checkpoint = (
        ROOT / f"study/artifacts/training/corrected-{family}-seed-{training_seed}/best.pt"
        if family in {"gnn", "endpoint", "candidate"}
        else None
    )
    model = None
    if checkpoint is not None:
        model, payload = load_model_checkpoint(checkpoint)
        if payload["model_family"] != family:
            raise ValueError("checkpoint family mismatch")
    output = ART / method / f"seed-{training_seed}" / f"n-{n}"
    contract = {
        "method": method,
        "seed": training_seed,
        "n": n,
        "episodes": protocol["episodes_per_cell"],
        "batch_size": protocol["batch_size"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": sha256_file(__file__),
        "parity_generator_sha256": sha256_file(ROOT / "experiments/parity_intervention.py"),
        "source_sha256": source_manifest()["sha256"],
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
    }
    fingerprint = digest(contract)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["contract_hash"] != fingerprint:
            raise ValueError("existing follow-up cell has a different contract")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest,
            {
                "status": "running",
                "contract": contract,
                "contract_hash": fingerprint,
                "environment": environment_metadata(torch.device("cpu")),
            },
        )
    chunk_paths = []
    for start in range(0, protocol["episodes_per_cell"], protocol["batch_size"]):
        path = output / f"chunk-{start:06}.json.gz"
        chunk_paths.append(path)
        count = min(protocol["batch_size"], protocol["episodes_per_cell"] - start)
        if path.exists():
            with gzip.open(path, "rt") as handle:
                saved = json.load(handle)
            if saved["contract_hash"] != fingerprint or len(saved["records"]) != count:
                raise ValueError("saved follow-up chunk contract mismatch")
            if sha256_file(path) != json.loads(manifest.read_text()).get("chunks", {}).get(
                path.name, sha256_file(path)
            ):
                raise ValueError("saved follow-up chunk hash mismatch")
            continue
        seeds = [
            (201 + training_seed) * 1_000_000 + n * 10_000 + e for e in range(start, start + count)
        ]
        try:
            begun = time.perf_counter()
            trajectories = (
                generate_balance(n, seeds)
                if method == "balance_parity"
                else generate_parity(model, method, n, seeds)
            )
            elapsed = time.perf_counter() - begun
            records = []
            for episode, episode_seed, trajectory in zip(
                range(start, start + count), seeds, trajectories, strict=True
            ):
                graph = graph_to_dict(trajectory.final_state)
                if not all(independent_verify(trajectory.final_state).values()):
                    raise ValueError("invalid generated graph")
                row = {
                    "method": method,
                    "seed": training_seed,
                    "n": n,
                    "episode": episode,
                    "episode_seed": episode_seed,
                    "record_id": f"{method}/s{training_seed}/n{n}/e{episode}",
                    "edge_count": trajectory.final_state.edge_count,
                    "optimal_edge_count": n * n // 4,
                    "optimality_ratio": trajectory.optimality_ratio,
                    "absolute_gap": n * n // 4 - trajectory.final_state.edge_count,
                    "exact_optimum": int(trajectory.optimality_ratio == 1),
                    "inference_time_seconds": elapsed / count,
                    "graph_sha256": digest(graph),
                    "checkpoint_sha256": contract["checkpoint_sha256"],
                    "parameter_count": sum(p.numel() for p in model.parameters()) if model else 0,
                    **trajectory_diagnostics(trajectory),
                }
                records.append(
                    {"row": row, "graph": graph, "actions": trajectory.actions, "probe": None}
                )
            temporary = path.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wt") as handle:
                json.dump(
                    {"contract_hash": fingerprint, "records": records},
                    handle,
                    separators=(",", ":"),
                )
            temporary.replace(path)
            with gzip.open(path, "rt") as handle:
                persisted = json.load(handle)
            for record in persisted["records"]:
                if not all(independent_verify(graph_from_dict(record["graph"])).values()):
                    raise ValueError("invalid persisted graph")
        except Exception as error:
            write_json(
                output / "failure.json",
                {"status": "failed", "error": repr(error), "chunk_start": start},
            )
            raise
    completion = json.loads(manifest.read_text())
    completion.update(
        status="complete",
        rows=protocol["episodes_per_cell"],
        chunks={p.name: sha256_file(p) for p in chunk_paths},
    )
    write_json(manifest, completion)
    return str(output.relative_to(ROOT))


def verify(protocol: dict) -> dict:
    jobs = [
        (m, s, n)
        for m in protocol["methods"]
        for s in protocol["training_seeds"]
        for n in protocol["sizes"]
    ]
    for method, seed, n in jobs:
        manifest = ART / method / f"seed-{seed}" / f"n-{n}/manifest.json"
        if not manifest.exists():
            raise ValueError(f"missing planned follow-up cell: {manifest}")
        value = json.loads(manifest.read_text())
        contract = value["contract"]
        if (contract["method"], contract["seed"], contract["n"]) != (method, seed, n):
            raise ValueError("follow-up cell identity mismatch")
        if contract["protocol_sha256"] != sha256_file(PROTOCOL):
            raise ValueError("follow-up protocol changed")
        if contract["script_sha256"] != sha256_file(__file__):
            raise ValueError("follow-up generator changed")
        if contract["parity_generator_sha256"] != sha256_file(
            ROOT / "experiments/parity_intervention.py"
        ):
            raise ValueError("frozen parity generator changed")
        if contract["source_sha256"] != source_manifest()["sha256"]:
            raise ValueError("follow-up inference source changed")
        family = method.removesuffix("_parity")
        checkpoint = (
            ROOT / f"study/artifacts/training/corrected-{family}-seed-{seed}/best.pt"
            if family in {"gnn", "endpoint", "candidate"}
            else None
        )
        if contract["checkpoint_sha256"] != (sha256_file(checkpoint) if checkpoint else None):
            raise ValueError("frozen checkpoint changed")
    result = verify_bundle(ART)
    if (
        result["cells"] != len(jobs)
        or result["graphs"] != len(jobs) * protocol["episodes_per_cell"]
    ):
        raise ValueError("incomplete follow-up matrix")
    write_json(STUDY / "verification.json", result)
    collect(ART, ART / "evaluation.csv")
    return result


def pilot() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = []
    for method in ["balance_parity", "gnn_parity", "endpoint_parity", "candidate_parity"]:
        model = None
        family = method.removesuffix("_parity")
        if family != "balance":
            model, _ = load_model_checkpoint(
                ROOT / f"study/artifacts/training/corrected-{family}-seed-0/best.pt"
            )
        seeds = [9_300_000 + i for i in range(4)]
        begun = time.perf_counter()
        if method == "balance_parity":
            generate_balance(40, seeds)
        else:
            generate_parity(model, method, 40, seeds)
        results.append(
            {
                "method": method,
                "n": 40,
                "episodes": len(seeds),
                "seconds_per_episode": (time.perf_counter() - begun) / len(seeds),
            }
        )
    write_json(STUDY / "resource_profile.json", {"results": results, "timing_seeds_excluded": True})
    print(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "run", "verify"])
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.stage == "pilot":
        pilot()
    elif args.stage == "verify":
        print(verify(protocol))
    else:
        jobs = [
            (m, s, n)
            for m in protocol["methods"]
            for s in protocol["training_seeds"]
            for n in protocol["sizes"]
        ]
        append_jsonl(
            STUDY / "ledger.jsonl",
            {
                "event": "launch",
                "time": time.time(),
                "jobs": len(jobs),
                "protocol_sha256": sha256_file(PROTOCOL),
            },
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
            for index, result in enumerate(pool.map(cell, jobs), 1):
                append_jsonl(
                    STUDY / "ledger.jsonl",
                    {"event": "complete", "time": time.time(), "cell": result},
                )
                print(f"follow-up {index}/{len(jobs)}", flush=True)
        print(verify(protocol))


if __name__ == "__main__":
    main()

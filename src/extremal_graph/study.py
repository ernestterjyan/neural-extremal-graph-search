"""Versioned, resumable research evaluation with persisted graph evidence.

Run ``python -m extremal_graph.study --help``. Completed chunks are atomic,
content-checked, and tied to a source/config/checkpoint contract on resume.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from .baselines import LeastDegreePolicy, LookaheadPolicy, UniformRandomPolicy, run_baseline_episode
from .config import ModelConfig
from .diagnostics import balanced_completion, preserving_actions, trajectory_diagnostics
from .features import collate_graph_states, legal_edges_from_state
from .graph import GraphState, Trajectory
from .policy import action_probabilities
from .rollouts import run_episode_batch
from .serialization import graph_from_dict, graph_to_dict
from .training import build_model, load_model_checkpoint
from .turan import construct_turan, turan_edge_count
from .utils import environment_metadata, set_global_seed, sha256_file, source_manifest, write_json


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def independent_verify(graph: GraphState) -> dict:
    """NetworkX implementation independent of environment and package verifier."""
    g = nx.Graph()
    g.add_nodes_from(range(graph.n))
    g.add_edges_from(graph.edges)
    triangle_free = not any(nx.triangles(g).values())
    maximal = triangle_free and all(set(g[u]) & set(g[v]) for u, v in nx.non_edges(g))
    return {"triangle_free": triangle_free, "maximal": maximal}


def structural_groups(state: GraphState, legal: list, layers: int, family: str) -> list:
    """Sufficient symbolic equivalence for this encoder, not equality of fitted logits.

    Mean-neighborhood signatures retain normalized color multiplicities. Equal
    signatures imply equal messages for any weights; the converse is not claimed.
    """
    import math

    adjacency = state.adjacency_sets()
    degree = state.degrees()
    incident = [0] * state.n
    for u, v in legal:
        incident[u] += 1
        incident[v] += 1
    signatures = [(degree[u], incident[u]) for u in range(state.n)]
    for _ in range(layers if family == "gnn" else 0):
        labels = {x: i for i, x in enumerate(sorted(set(signatures)))}
        colors = [labels[x] for x in signatures]
        next_signatures = []
        for u in range(state.n):
            counts = {}
            for v in adjacency[u]:
                counts[colors[v]] = counts.get(colors[v], 0) + 1
            normalized = []
            for color, count in sorted(counts.items()):
                divisor = math.gcd(count, degree[u])
                normalized.append((color, count // divisor, degree[u] // divisor))
            next_signatures.append((colors[u], tuple(normalized)))
        signatures = next_signatures
    labels = {x: i for i, x in enumerate(sorted(set(signatures)))}
    colors = [labels[x] for x in signatures]
    result = []
    for u, v in legal:
        statistics = (
            degree[u] + degree[v],
            abs(degree[u] - degree[v]),
            incident[u] + incident[v],
            abs(incident[u] - incident[v]),
        )
        result.append(
            statistics
            if family == "candidate"
            else (min(colors[u], colors[v]), max(colors[u], colors[v]), statistics)
        )
    return result


def probe_trajectory(model, trajectory: Trajectory, family: str, layers: int) -> dict:
    """Prespecified diagnostic subset, evaluated separately from timed generation."""
    counts = {
        "feasible_decisions": 0,
        "mixed_group_states": 0,
        "unavoidable_alias_states": 0,
        "sum_bad_action_mass": 0.0,
        "sum_bad_mass_in_mixed_groups": 0.0,
        "chosen_bad_action": 0,
        "chosen_bad_in_mixed_group": 0,
    }
    first_mass = None
    decision_records = []
    with torch.inference_mode():
        for step, (state, action) in enumerate(
            zip(trajectory.states, trajectory.actions, strict=True), 1
        ):
            if not balanced_completion(state)[0]:
                break
            legal = legal_edges_from_state(state)
            good = preserving_actions(state, legal)
            batch = collate_graph_states([state], [legal])
            probabilities, _ = action_probabilities(
                model(batch), candidate_mask=batch.candidate_mask
            )
            values = probabilities[0, : len(legal)].tolist()
            bad_mass = sum(p for p, keeps in zip(values, good, strict=True) if not keeps)
            counts["feasible_decisions"] += 1
            counts["sum_bad_action_mass"] += bad_mass
            chosen = legal.index(action)
            counts["chosen_bad_action"] += int(not good[chosen])
            mixed = set()
            structural_bound = None
            if family in {"gnn", "endpoint", "candidate"}:
                groups = structural_groups(state, legal, layers, family)
                grouped = {}
                for i, key in enumerate(groups):
                    grouped.setdefault(key, []).append(i)
                mixed = {
                    i
                    for indices in grouped.values()
                    if any(good[j] for j in indices) and not all(good[j] for j in indices)
                    for i in indices
                }
                best_fraction = max(
                    sum(good[i] for i in ids) / len(ids) for ids in grouped.values()
                )
                structural_bound = 1 - best_fraction
                counts["mixed_group_states"] += int(bool(mixed))
                counts["unavoidable_alias_states"] += int(structural_bound > 0)
                counts["sum_bad_mass_in_mixed_groups"] += sum(
                    values[i] for i in mixed if not good[i]
                )
                counts["chosen_bad_in_mixed_group"] += int(chosen in mixed and not good[chosen])
            if not good[chosen]:
                first_mass = bad_mass
            decision_records.append(
                {
                    "step": step,
                    "bad_action_mass": bad_mass,
                    "chosen_destroys_completion": not good[chosen],
                    "mixed_alias_group_present": bool(mixed),
                    "structural_bad_mass_lower_bound": structural_bound,
                }
            )
    return counts | {"first_loss_bad_mass": first_mass, "decisions": decision_records}


def evaluate_cell(
    *,
    output: str | Path,
    method: str,
    seed: int,
    n: int,
    episodes: int,
    checkpoint: str | Path | None = None,
    batch_size: int = 32,
    sampling_protocol: str = "episode-v2",
    diagnostic_episodes: int = 0,
    diagnostic_sizes: tuple = (14, 24, 40),
    temperature: float = 1.0,
    stop_after_chunks: int | None = None,
) -> Path:
    """Resume a cell without reusing results from an incompatible experiment."""
    if episodes < 1 or batch_size < 1 or n < 2 or diagnostic_episodes < 0:
        raise ValueError("invalid evaluation counts")
    torch.set_num_threads(1)
    set_global_seed(seed)
    model = None
    model_config = None
    model_hash = None
    if checkpoint is not None:
        model, payload = load_model_checkpoint(checkpoint)
        model_config = ModelConfig(**payload["model_config"])
        if model_config.family != method:
            raise ValueError("checkpoint family does not match method")
        model_hash = sha256_file(checkpoint)
    elif method.startswith("untrained_"):
        family = method.removeprefix("untrained_")
        model_config = ModelConfig(family=family, hidden_dim=67 if family == "mlp" else 64)
        model = build_model(model_config).eval()
    elif method in {"gnn", "mlp", "candidate", "endpoint"}:
        raise ValueError("trained method requires checkpoint")
    policies = {
        "random": UniformRandomPolicy,
        "least_degree": LeastDegreePolicy,
        "lookahead": LookaheadPolicy,
    }
    if model is None and method not in {*policies, "turan_oracle"}:
        raise ValueError("unknown method")
    if model_config is not None and model_config.family == "mlp" and n > model_config.max_nodes:
        raise ValueError("MLP size unsupported; do not report as a scored failure")
    cell = Path(output) / method / f"seed-{seed}" / f"n-{n}"
    contract = {
        "method": method,
        "seed": seed,
        "n": n,
        "episodes": episodes,
        "checkpoint_sha256": model_hash,
        "model_config": asdict(model_config) if model_config else None,
        "batch_size": batch_size,
        "sampling_protocol": sampling_protocol,
        "temperature": temperature,
        "diagnostic_episodes": diagnostic_episodes,
        "diagnostic_sizes": list(diagnostic_sizes),
        "source_sha256": source_manifest()["sha256"],
        "seed_rule": "(seed+1)*1000000+n*10000+episode",
    }
    fingerprint = digest(contract)
    manifest_path = cell / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["contract_hash"] != fingerprint:
            raise ValueError("evaluation resume contract mismatch")
    else:
        cell.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest_path,
            {
                "status": "running",
                "contract": contract,
                "contract_hash": fingerprint,
                "environment": environment_metadata(torch.device("cpu")),
                "checkpoint_path": str(checkpoint) if checkpoint else None,
            },
        )
    completed_chunks = 0
    expected_chunks = []
    for start in range(0, episodes, batch_size):
        path = cell / f"chunk-{start:06}.json.gz"
        expected_chunks.append(path)
        count = min(batch_size, episodes - start)
        if path.exists():
            with gzip.open(path, "rt") as handle:
                saved = json.load(handle)
            if saved["contract_hash"] != fingerprint or len(saved["records"]) != count:
                raise ValueError("invalid persisted chunk")
            for record in saved["records"]:
                if digest(record["graph"]) != record["row"]["graph_sha256"]:
                    raise ValueError("persisted graph hash mismatch")
            continue
        seeds = [(seed + 1) * 1_000_000 + n * 10_000 + i for i in range(start, start + count)]
        try:
            begun = time.perf_counter()
            if model is not None:
                trajectories = run_episode_batch(
                    model,
                    n=n,
                    seeds=seeds,
                    temperature=temperature,
                    sampling_protocol=sampling_protocol,
                )
            elif method == "turan_oracle":
                graph = construct_turan(n, 2)
                trajectories = [Trajectory(n, 2, (), (), (), graph, 1.0) for _ in seeds]
            else:
                trajectories = [run_baseline_episode(n, policies[method](), seed=s) for s in seeds]
            elapsed = time.perf_counter() - begun
            records = []
            for offset, trajectory in enumerate(trajectories):
                episode = start + offset
                graph = graph_to_dict(trajectory.final_state)
                verification = independent_verify(trajectory.final_state)
                if not all(verification.values()):
                    raise RuntimeError("independent verification failed")
                diagnostic = trajectory_diagnostics(trajectory)
                if method == "turan_oracle":
                    diagnostic.update(
                        first_loss_step=None, first_loss_cause=None, first_odd_cycle_step=None
                    )
                row = {
                    "method": method,
                    "seed": seed,
                    "n": n,
                    "episode": episode,
                    "episode_seed": seeds[offset],
                    "record_id": f"{method}/s{seed}/n{n}/e{episode}",
                    "edge_count": trajectory.final_state.edge_count,
                    "optimal_edge_count": turan_edge_count(n, 2),
                    "optimality_ratio": trajectory.optimality_ratio,
                    "absolute_gap": turan_edge_count(n, 2) - trajectory.final_state.edge_count,
                    "exact_optimum": int(trajectory.optimality_ratio == 1.0),
                    "constraint_violations": 0,
                    "terminal_maximal": 1,
                    "inference_time_seconds": elapsed / count,
                    "generation_batch_size": count,
                    "batch_start": start,
                    "graph_sha256": digest(graph),
                    "checkpoint_sha256": model_hash,
                    "parameter_count": sum(p.numel() for p in model.parameters()) if model else 0,
                    "status": "ok",
                    **diagnostic,
                }
                probe = None
                if model is not None and episode < diagnostic_episodes and n in diagnostic_sizes:
                    probe = probe_trajectory(
                        model, trajectory, model_config.family, model_config.message_passing_layers
                    )
                records.append(
                    {"row": row, "graph": graph, "actions": trajectory.actions, "probe": probe}
                )
            temporary = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(temporary, "wt") as handle:
                json.dump(
                    {"contract_hash": fingerprint, "records": records},
                    handle,
                    separators=(",", ":"),
                )
            temporary.replace(path)
            # Verify the objects actually persisted, not just the in-memory originals.
            with gzip.open(path, "rt") as handle:
                persisted = json.load(handle)
            for record in persisted["records"]:
                if not all(independent_verify(graph_from_dict(record["graph"])).values()):
                    raise RuntimeError("persisted graph verification failed")
        except Exception as error:
            write_json(
                cell / "failure.json",
                {
                    "status": "numerical_failure"
                    if isinstance(error, FloatingPointError)
                    else "failed",
                    "error": repr(error),
                    "start": start,
                    "time": time.time(),
                },
            )
            raise
        completed_chunks += 1
        if stop_after_chunks is not None and completed_chunks >= stop_after_chunks:
            return cell
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        status="complete",
        completed_at=time.time(),
        chunks={p.name: sha256_file(p) for p in expected_chunks},
        rows=episodes,
    )
    write_json(manifest_path, manifest)
    return cell


def verify_bundle(root: str | Path) -> dict:
    """Check every completed cell, hashes, unique identities, math, and saved graphs."""
    root = Path(root)
    count = 0
    cells = 0
    identities = set()
    for manifest_path in sorted(root.glob("*/seed-*/n-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete cell: {manifest_path}")
        contract = manifest["contract"]
        if digest(contract) != manifest["contract_hash"]:
            raise ValueError("manifest contract hash mismatch")
        seen = []
        for filename, checksum in manifest["chunks"].items():
            path = manifest_path.parent / filename
            if sha256_file(path) != checksum:
                raise ValueError("chunk content hash mismatch")
            with gzip.open(path, "rt") as handle:
                chunk = json.load(handle)
            if chunk["contract_hash"] != manifest["contract_hash"]:
                raise ValueError("chunk contract mismatch")
            for record in chunk["records"]:
                row = record["row"]
                graph = graph_from_dict(record["graph"])
                if row["record_id"] in identities:
                    raise ValueError("duplicate result identity")
                identities.add(row["record_id"])
                if digest(record["graph"]) != row["graph_sha256"]:
                    raise ValueError("graph hash mismatch")
                if not all(independent_verify(graph).values()):
                    raise ValueError("invalid persisted graph")
                optimum = turan_edge_count(graph.n, 2)
                if row["edge_count"] != graph.edge_count or row["optimal_edge_count"] != optimum:
                    raise ValueError("incorrect edge count")
                if abs(row["optimality_ratio"] - graph.edge_count / optimum) > 1e-12:
                    raise ValueError("incorrect ratio")
                if row["exact_optimum"] != int(graph.edge_count == optimum):
                    raise ValueError("incorrect exact flag")
                if record["actions"] and sorted(map(tuple, record["actions"])) != list(graph.edges):
                    raise ValueError("action/final graph mismatch")
                seen.append(row["episode"])
                count += 1
        if sorted(seen) != list(range(contract["episodes"])):
            raise ValueError("missing or repeated episode")
        cells += 1
    if not cells:
        raise ValueError("no completed evaluation cells")
    return {"cells": cells, "graphs": count, "valid": True}


def collect(root: str | Path, output: str | Path) -> Path:
    rows = []
    for path in sorted(Path(root).glob("*/seed-*/n-*/chunk-*.json.gz")):
        with gzip.open(path, "rt") as handle:
            rows.extend(record["row"] for record in json.load(handle)["records"])
    frame = pd.DataFrame(rows)
    if frame.empty or frame.record_id.duplicated().any():
        raise ValueError("missing or duplicate records")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("evaluate")
    for name in ["output", "method"]:
        cell.add_argument("--" + name, required=True)
    for name in ["seed", "n", "episodes"]:
        cell.add_argument("--" + name, type=int, required=True)
    cell.add_argument("--checkpoint")
    cell.add_argument("--batch-size", type=int, default=32)
    cell.add_argument("--sampling-protocol", default="episode-v2")
    cell.add_argument("--diagnostic-episodes", type=int, default=10)
    verify = commands.add_parser("verify")
    verify.add_argument("root")
    combine = commands.add_parser("collect")
    combine.add_argument("root")
    combine.add_argument("output")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "evaluate":
        print(evaluate_cell(**args))
    elif command == "verify":
        print(json.dumps(verify_bundle(**args)))
    else:
        print(collect(**args))


if __name__ == "__main__":
    main()

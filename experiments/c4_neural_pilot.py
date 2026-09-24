"""Small fixed-budget neural C4-free construction study; no legacy model is reused."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.c4_solver_pilot import (  # noqa: E402
    add,
    edge_count,
    edges,
    legal_edges,
    verify_graph,
)
from extremal_graph.features import GraphTensorBatch  # noqa: E402
from extremal_graph.policy import EdgePolicy, sample_actions  # noqa: E402

STUDY = ROOT / "study/c4_neural"
PROTOCOL = STUDY / "protocol.json"
ART = STUDY / "artifacts"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract() -> dict:
    return {"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__))}


def collate(items: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]]) -> GraphTensorBatch:
    """C4-specific features with the original symmetric edge-policy tensor API."""
    if not items:
        raise ValueError("an empty C4 batch has no actions")
    size = len(items)
    nmax = max(len(adjacency) for adjacency, _ in items)
    cmax = max(len(candidates) for _, candidates in items)
    adjacency_tensor = torch.zeros((size, nmax, nmax), dtype=torch.float32)
    node_features = torch.zeros((size, nmax, 4), dtype=torch.float32)
    node_mask = torch.zeros((size, nmax), dtype=torch.bool)
    candidates_tensor = torch.zeros((size, cmax, 2), dtype=torch.long)
    candidate_features = torch.zeros((size, cmax, 5), dtype=torch.float32)
    candidate_mask = torch.zeros((size, cmax), dtype=torch.bool)
    for row, (adjacency, candidates) in enumerate(items):
        n = len(adjacency)
        node_mask[row, :n] = True
        degrees = [bits.bit_count() for bits in adjacency]
        incidence = [0] * n
        for u, v in candidates:
            incidence[u] += 1
            incidence[v] += 1
        degree_scale = max(1, n - 1)
        upper = n * (1 + math.sqrt(4 * n - 3)) / 4
        progress = (sum(degrees) / 2) / upper
        for u, bits in enumerate(adjacency):
            for v in range(n):
                if bits & (1 << v):
                    adjacency_tensor[row, u, v] = 1
            two_hop = 0
            neighbors = bits
            while neighbors:
                bit = neighbors & -neighbors
                two_hop |= adjacency[bit.bit_length() - 1]
                neighbors -= bit
            node_features[row, u] = torch.tensor(
                [
                    degrees[u] / degree_scale,
                    incidence[u] / degree_scale,
                    progress,
                    (two_hop & ~(1 << u)).bit_count() / degree_scale,
                ]
            )
        for index, (u, v) in enumerate(candidates):
            candidates_tensor[row, index] = torch.tensor((u, v))
            candidate_mask[row, index] = True
            candidate_features[row, index] = torch.tensor(
                [
                    (degrees[u] + degrees[v]) / (2 * degree_scale),
                    abs(degrees[u] - degrees[v]) / degree_scale,
                    (incidence[u] + incidence[v]) / (2 * degree_scale),
                    abs(incidence[u] - incidence[v]) / degree_scale,
                    (adjacency[u] & adjacency[v]).bit_count() / max(1, n - 2),
                ]
            )
    return GraphTensorBatch(
        adjacency_tensor,
        node_features,
        node_mask,
        candidates_tensor,
        candidate_features,
        candidate_mask,
    )


def model_for(family: str) -> EdgePolicy:
    return EdgePolicy(
        node_feature_dim=4,
        candidate_feature_dim=5,
        hidden_dim=64,
        message_passing_layers=3 if family == "gnn" else 0,
    )


def generate(model: nn.Module, n: int, seeds: list[int], *, trace: bool = False) -> list[dict]:
    adjacency = [[0] * n for _ in seeds]
    episode_rngs = [torch.Generator().manual_seed(seed) for seed in seeds]
    traces: list[list[tuple[tuple[int, ...], tuple[tuple[int, int], ...], int]]] = [
        [] for _ in seeds
    ]
    active = list(range(len(seeds)))
    model.eval()
    with torch.inference_mode():
        while active:
            rows = []
            available = []
            still_active = []
            for episode in active:
                candidates = tuple(legal_edges(adjacency[episode]))
                if candidates:
                    rows.append((tuple(adjacency[episode]), candidates))
                    available.append(candidates)
                    still_active.append(episode)
            if not still_active:
                break
            batch = collate(rows)
            chosen = sample_actions(
                model(batch),
                candidate_mask=batch.candidate_mask,
                generators=[episode_rngs[episode] for episode in still_active],
            ).tolist()
            for episode, row, candidates, index in zip(
                still_active, rows, available, chosen, strict=True
            ):
                if trace:
                    traces[episode].append((row[0], candidates, index))
                add(adjacency[episode], candidates[index])
            active = still_active
    return [
        {"adjacency": value, "edge_count": edge_count(value), "trace": traces[index]}
        for index, value in enumerate(adjacency)
    ]


def _sample_batch(model: nn.Module, n: int, seeds: list[int], *, trace: bool) -> list[dict]:
    outputs = []
    for start in range(0, len(seeds), 32):
        outputs.extend(generate(model, n, seeds[start : start + 32], trace=trace))
    return outputs


def train_one(family: str, seed: int, protocol: dict) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = ART / "training" / f"{family}-seed-{seed}"
    checkpoint = folder / "best.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["contract"] != contract():
            raise ValueError("existing C4 neural checkpoint has incompatible source")
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(10000 + seed)
    model = model_for(family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    selection_rng = random.Random(30000 + seed)
    metrics = []
    best_score = -1.0
    best_step = 0
    best_state = None
    n = protocol["training_size"]
    for iteration in range(1, protocol["iterations"] + 1):
        begun = time.perf_counter()
        train_seeds = [
            (500 + seed) * 1_000_000 + iteration * 1_000 + episode
            for episode in range(protocol["rollouts_per_iteration"])
        ]
        rollout = _sample_batch(model, n, train_seeds, trace=True)
        order = list(range(len(rollout)))
        selection_rng.shuffle(order)
        order.sort(key=lambda index: rollout[index]["edge_count"], reverse=True)
        pairs = [
            pair
            for index in order[: protocol["elite_trajectories"]]
            for pair in rollout[index]["trace"][: protocol["elite_prefix_length"]]
        ]
        if len(pairs) != protocol["elite_trajectories"] * protocol["elite_prefix_length"]:
            raise ValueError("an elite rollout terminated before the selected prefix")
        model.train()
        losses = []
        for _ in range(protocol["updates_per_iteration"]):
            examples = [
                pairs[selection_rng.randrange(len(pairs))]
                for _ in range(protocol["optimizer_batch_size"])
            ]
            batch = collate([(adjacency, candidates) for adjacency, candidates, _ in examples])
            targets = torch.tensor([index for _, _, index in examples], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite C4 neural loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_score = None
        if iteration % protocol["validation_interval"] == 0:
            validation_seeds = [
                (600 + seed) * 1_000_000 + iteration * 1_000 + episode
                for episode in range(protocol["validation_episodes"])
            ]
            validation = _sample_batch(model, n, validation_seeds, trace=False)
            validation_score = sum(row["edge_count"] for row in validation) / len(validation)
            if validation_score > best_score:
                best_score = validation_score
                best_step = iteration
                best_state = copy.deepcopy(model.state_dict())
        metrics.append(
            {
                "iteration": iteration,
                "mean_train_edges": sum(row["edge_count"] for row in rollout) / len(rollout),
                "validation_mean_edges": validation_score,
                "mean_loss": sum(losses) / len(losses),
                "seconds": time.perf_counter() - begun,
            }
        )
        if iteration % 10 == 0:
            print(f"train {family} seed={seed} iteration={iteration}", flush=True)
    if best_state is None:
        raise ValueError("no validation checkpoint selected")
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        {
            "family": family,
            "seed": seed,
            "contract": contract(),
            "best_step": best_step,
            "best_validation_score": best_score,
            "model_state": best_state,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        temporary,
    )
    temporary.replace(checkpoint)
    (folder / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics)
    )


def evaluate_one(family: str, seed: int, n: int, protocol: dict) -> list[dict]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = ART / "training" / f"{family}-seed-{seed}/best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["contract"] != contract():
        raise ValueError("C4 neural evaluation source mismatch")
    model = model_for(family)
    model.load_state_dict(payload["model_state"])
    seeds = [
        (700 + seed) * 1_000_000 + n * 10_000 + episode
        for episode in range(protocol["evaluation_episodes_per_cell"])
    ]
    begun = time.perf_counter()
    outputs = _sample_batch(model, n, seeds, trace=False)
    elapsed = (time.perf_counter() - begun) / len(outputs)
    records = []
    for episode, output in enumerate(outputs):
        edge_list = edges(output["adjacency"])
        verify_graph(n, edge_list)
        records.append(
            {
                "family": family,
                "training_seed": seed,
                "n": n,
                "episode": episode,
                "episode_seed": seeds[episode],
                "edge_count": output["edge_count"],
                "edges": edge_list,
                "seconds_per_graph": elapsed,
                "checkpoint_sha256": sha(checkpoint),
                "contract": contract(),
            }
        )
    return records


def evaluate(protocol: dict) -> None:
    results = STUDY / "results.jsonl"
    if results.exists():
        raise FileExistsError("C4 neural evaluation already exists; do not replace draws")
    jobs = [
        (family, seed, n)
        for family in protocol["families"]
        for seed in protocol["training_seeds"]
        for n in protocol["evaluation_sizes"]
    ]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(evaluate_one, family, seed, n, protocol) for family, seed, n in jobs]
        for (family, seed, n), future in zip(jobs, futures, strict=True):
            records.extend(future.result())
            print(f"evaluate {family} seed={seed} n={n}", flush=True)
    temporary = results.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    temporary.replace(results)


def verify(protocol: dict) -> list[dict]:
    for family in protocol["families"]:
        for seed in protocol["training_seeds"]:
            folder = ART / "training" / f"{family}-seed-{seed}"
            payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            if payload["contract"] != contract():
                raise ValueError("C4 neural training contract mismatch")
            metrics = [
                json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
            ]
            if len(metrics) != protocol["iterations"]:
                raise ValueError("C4 neural training log incomplete")
    records = [json.loads(line) for line in (STUDY / "results.jsonl").read_text().splitlines()]
    expected = {
        (family, seed, n, episode)
        for family in protocol["families"]
        for seed in protocol["training_seeds"]
        for n in protocol["evaluation_sizes"]
        for episode in range(protocol["evaluation_episodes_per_cell"])
    }
    if (
        len(records) != len(expected)
        or {(row["family"], row["training_seed"], row["n"], row["episode"]) for row in records}
        != expected
    ):
        raise ValueError("C4 neural evaluation missing or duplicated")
    for record in records:
        if record["contract"] != contract():
            raise ValueError("C4 neural evaluation contract mismatch")
        if record["edge_count"] != len(record["edges"]):
            raise ValueError("C4 neural edge count mismatch")
        if record["edge_count"] > protocol["known_exact_values"][str(record["n"])]:
            raise ValueError("C4 neural result exceeds the published exact value")
        expected_seed = (
            (700 + record["training_seed"]) * 1_000_000 + record["n"] * 10_000 + record["episode"]
        )
        if record["episode_seed"] != expected_seed:
            raise ValueError("C4 neural evaluation seed mismatch")
        checkpoint = ART / "training" / f"{record['family']}-seed-{record['training_seed']}/best.pt"
        if record["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("C4 neural checkpoint mismatch")
        verify_graph(record["n"], record["edges"])
    return records


def report(protocol: dict) -> None:
    records = verify(protocol)
    rows = []
    for n in protocol["evaluation_sizes"]:
        for family in protocol["families"]:
            subset = [r for r in records if r["n"] == n and r["family"] == family]
            values = [r["edge_count"] for r in subset]
            reference = protocol["known_exact_values"][str(n)]
            rows.append(
                {
                    "n": n,
                    "family": family,
                    "known_exact": reference,
                    "mean_edges": sum(values) / len(values),
                    "max_edges": max(values),
                    "mean_gap": reference - sum(values) / len(values),
                    "exact_runs": sum(value == reference for value in values),
                    "graphs": len(values),
                    "mean_seconds": sum(r["seconds_per_graph"] for r in subset) / len(subset),
                }
            )
    with (STUDY / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print((STUDY / "summary.csv").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "verify", "report"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads(PROTOCOL.read_text())
    if args.stage == "train":
        jobs = [
            (family, seed) for seed in protocol["training_seeds"] for family in protocol["families"]
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
            futures = [pool.submit(train_one, family, seed, protocol) for family, seed in jobs]
            for (family, seed), future in zip(jobs, futures, strict=True):
                future.result()
                print(f"complete {family} seed={seed}", flush=True)
    elif args.stage == "evaluate":
        evaluate(protocol)
    elif args.stage == "verify":
        print({"graphs": len(verify(protocol)), "valid": True})
    else:
        report(protocol)


if __name__ == "__main__":
    main()

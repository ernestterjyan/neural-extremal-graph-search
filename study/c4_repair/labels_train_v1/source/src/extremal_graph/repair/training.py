"""Supervised region ranking; independent model seeds, common labelled data."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from .evidence import append_record, load_records, sha, verify_contract
from .graph import Graph, verify_witness
from .regions import Region
from .selectors import features, make_model


def labelled_states(directory: Path):
    contract = verify_contract(directory)
    if contract["protocol"]["stage"] != "collection":
        raise ValueError("training requires labelled collection evidence")
    examples = []
    records = load_records(directory / "results.jsonl")
    expected = {
        (n, i)
        for n in contract["protocol"]["sizes"]
        for i in range(contract["protocol"]["states_per_size"])
    }
    if len(records) != len(expected) or {(r["n"], r["replication"]) for r in records} != expected:
        raise ValueError("complete labelled state panel required")
    for record in records:
        if (
            record["source_sha256"] != contract["source_sha256"]
            or record["protocol_sha256"] != contract["protocol_sha256"]
        ):
            raise ValueError("label provenance mismatch")
        if record["status"] != "complete":
            raise ValueError("failed label collection cannot silently enter training")
        graph = Graph.from_edges(record["n"], record["initial_edges"])
        verify_witness(graph.n, graph.edges())
        if not record["repairs"]:
            raise ValueError("empty region label pool")
        for outcome in record["repairs"]:
            verify_witness(graph.n, outcome["edges"])
            result = Graph.from_edges(graph.n, outcome["edges"])
            if result.m - graph.m != outcome["gain"]:
                raise ValueError("label gain mismatch")
            outside = set(range(graph.n)) - set(outcome["region"]["vertices"])
            if any(graph.has(u, v) != result.has(u, v) for u in outside for v in outside if u < v):
                raise ValueError("label repair changed fixed complement")
        regions = [
            Region(tuple(r["region"]["vertices"]), r["region"]["family"]) for r in record["repairs"]
        ]
        adjacency, node, engineered = features(graph, regions)
        gains = torch.tensor([r["gain"] for r in record["repairs"]], dtype=torch.float32)
        examples.append((adjacency, node, engineered, regions, gains))
    return examples


def ranking_loss(model, example):
    adjacency, node, engineered, regions, gains = example
    scores = model(adjacency, node, engineered, regions)
    target = torch.softmax(gains, dim=0)
    return -(target * torch.log_softmax(scores, dim=0)).sum()


def train_family(
    train_directory: Path,
    validation_directory: Path,
    output: Path,
    *,
    family: str,
    seed: int,
    epochs: int = 30,
    learning_rate: float = 0.001,
) -> dict:
    if output.exists():
        raise FileExistsError("refusing to replace a trained model")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(seed)
    train = labelled_states(train_directory)
    validation = labelled_states(validation_directory)
    if not train or not validation or epochs < 1:
        raise ValueError("nonempty train/validation panels and positive epochs required")
    model = make_model(family)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train)))
        rng.shuffle(indices)
        losses = []
        for index in indices:
            loss = ranking_loss(model, train[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_loss = sum(ranking_loss(model, example).item() for example in validation) / len(
                validation
            )
        append_record(
            output / "epochs.jsonl",
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "validation_loss": val_loss,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(
                {
                    "family": family,
                    "architecture": model.architecture,
                    "seed": seed,
                    "state_dict": model.state_dict(),
                },
                output / "best.pt",
            )
    torch.save(
        {
            "family": family,
            "architecture": model.architecture,
            "seed": seed,
            "state_dict": model.state_dict(),
        },
        output / "last.pt",
    )
    metadata = {
        "family": family,
        "architecture": model.architecture,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "learning_rate": learning_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha(output / "best.pt"),
        "train_contract_sha256": sha(train_directory / "contract.json"),
        "validation_contract_sha256": sha(validation_directory / "contract.json"),
        "train_results_sha256": sha(train_directory / "results.jsonl"),
        "validation_results_sha256": sha(validation_directory / "results.jsonl"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_model(path: Path, expected_sha256: str):
    if sha(path) != expected_sha256:
        raise ValueError("checkpoint digest mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["family"])
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("checkpoint architecture differs from the retained model implementation")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model

"""Fixed-size, equal-rollout, equal-update training and held-out evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.config import load_training_config  # noqa: E402
from extremal_graph.features import (  # noqa: E402
    action_indices,
    collate_graph_states,
    legal_edges_from_state,
)
from extremal_graph.study import collect, evaluate_cell, verify_bundle  # noqa: E402
from extremal_graph.training import (  # noqa: E402
    _collect_rollouts,
    _validate,
    build_model,
    select_elite_trajectories,
)
from extremal_graph.utils import (  # noqa: E402
    append_jsonl,
    atomic_torch_save,
    capture_random_state,
    environment_metadata,
    restore_random_state,
    save_source_snapshot,
    set_global_seed,
    sha256_file,
    source_manifest,
    write_json,
)

STUDY = ROOT / "study/followup"
ART = STUDY / "artifacts"
PROTOCOL = STUDY / "matched_protocol.json"


def _contract(family: str, seed: int) -> dict:
    config_path = ROOT / f"experiments/configs/corrected-{family}.toml"
    model_config = load_training_config(config_path, seed).model
    return {
        "family": family,
        "seed": seed,
        "protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": sha256_file(__file__),
        "source_sha256": source_manifest()["sha256"],
        "model_config_sha256": sha256_file(config_path),
        "model_config": asdict(model_config),
    }


def _optimize(model, optimizer, pairs, rng, *, updates, batch_size):
    model.train()
    losses = []
    for _ in range(updates):
        batch_pairs = [pairs[rng.randrange(len(pairs))] for _ in range(batch_size)]
        states = [state for state, _ in batch_pairs]
        actions = [action for _, action in batch_pairs]
        candidates = [legal_edges_from_state(state) for state in states]
        batch = collate_graph_states(states, candidates)
        targets = action_indices(candidates, actions)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite matched-training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def train_cell(job: tuple[str, int]) -> str:
    family, seed = job
    protocol = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(protocol["threads_per_worker"])
    set_global_seed(1_000 + seed)
    contract = _contract(family, seed)
    output = ART / "matched_training" / f"{family}-seed-{seed}"
    latest = output / "latest.pt"
    completion = output / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["contract"] != contract or sha256_file(output / "best.pt") != saved["best_sha256"]:
            raise ValueError("existing matched training has an incompatible contract or checksum")
        return str(output)
    model = build_model(
        load_training_config(ROOT / f"experiments/configs/corrected-{family}.toml", seed).model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = random.Random(2_000 + seed)
    metrics = output / "metrics.jsonl"
    best_score = -math.inf
    best_step = 0
    best_state = None
    start_step = 1
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload["contract"] != contract:
            raise ValueError("matched-training resume contract mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.setstate(payload["selection_rng_state"])
        restore_random_state(payload["random_state"])
        best_score = payload["best_score"]
        best_step = payload["best_step"]
        best_state = payload["best_state"]
        start_step = payload["step"] + 1
        records = [json.loads(x) for x in metrics.read_text().splitlines()]
        records = [r for r in records if r["step"] <= payload["step"]]
        if len(records) == payload["step"] - 1:
            records.append(payload["last_metric"])
        if len(records) != payload["step"] or [r["step"] for r in records] != list(
            range(1, payload["step"] + 1)
        ):
            raise ValueError("matched-training metrics cannot be reconciled with checkpoint")
        metrics.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        append_jsonl(
            output / "resume_events.jsonl",
            {
                "time": time.time(),
                "checkpoint_sha256": sha256_file(latest),
                "next_step": start_step,
                "environment": environment_metadata(torch.device("cpu")),
            },
        )
    elif output.exists():
        raise ValueError("run directory exists without a resumable checkpoint")
    else:
        output.mkdir(parents=True)
        metrics.write_text("")
        snapshot = save_source_snapshot(output / "source.tar.gz")
        write_json(
            output / "metadata.json",
            {
                "contract": contract,
                "protocol": protocol,
                "environment": environment_metadata(torch.device("cpu")),
                "source_snapshot": snapshot,
                "started_at_unix": time.time(),
            },
        )

    for step in range(start_step, protocol["iterations"] + 1):
        begun = time.perf_counter()
        try:
            trajectories = _collect_rollouts(
                model,
                sizes=(protocol["training_size"],),
                count=protocol["rollouts_per_iteration"],
                seed=200 + seed,
                global_step=step,
                r=2,
                temperature=1.0,
                device=torch.device("cpu"),
                sampling_protocol="episode-v2",
            )
            elite = select_elite_trajectories(trajectories, protocol["elite_fraction"], rng)
            if len(elite) != protocol["elite_trajectories_per_iteration"]:
                raise ValueError("elite count changed")
            prefix = protocol["elite_prefix_length"]
            if any(len(trajectory.actions) < prefix for trajectory in elite):
                raise ValueError("a terminal trajectory is shorter than the fixed elite prefix")
            pairs = [
                (state, action)
                for trajectory in elite
                for state, action in zip(
                    trajectory.states[:prefix], trajectory.actions[:prefix], strict=True
                )
            ]
            loss = _optimize(
                model,
                optimizer,
                pairs,
                rng,
                updates=protocol["updates_per_iteration"],
                batch_size=protocol["batch_size"],
            )
            score = None
            if step % protocol["validation_interval"] == 0:
                score = _validate(
                    model,
                    sizes=(protocol["training_size"],),
                    episodes=protocol["validation_episodes"],
                    seed=200 + seed,
                    global_step=step,
                    r=2,
                    temperature=1.0,
                    device=torch.device("cpu"),
                )[protocol["training_size"]]
                if score > best_score:
                    best_score = score
                    best_step = step
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
            record = {
                "step": step,
                "family": family,
                "seed": seed,
                "training_rollouts": protocol["rollouts_per_iteration"],
                "training_actions": sum(len(t.actions) for t in trajectories),
                "elite_trajectories": len(elite),
                "elite_prefix_pairs": len(pairs),
                "optimizer_updates": protocol["updates_per_iteration"],
                "optimizer_pair_presentations": protocol["updates_per_iteration"]
                * protocol["batch_size"],
                "validation_episodes": protocol["validation_episodes"] if score is not None else 0,
                "validation_score": score,
                "best_validation_score": best_score if best_step else None,
                "best_step": best_step,
                "loss": loss,
                "seconds": time.perf_counter() - begun,
            }
            checkpoint = {
                "format_version": 1,
                "contract": contract,
                "step": step,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                "selection_rng_state": rng.getstate(),
                "random_state": capture_random_state(),
                "best_score": best_score,
                "best_step": best_step,
                "best_state": best_state,
                "last_metric": record,
            }
            atomic_torch_save(checkpoint, latest)
            append_jsonl(metrics, record)
        except Exception as error:
            write_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "step": step,
                    "error": repr(error),
                    "time": time.time(),
                },
            )
            raise
    if best_state is None:
        raise ValueError("matched-training run ended without a selected validation checkpoint")
    selected = {
        "format_version": 1,
        "model_state": best_state,
        "model_config": contract["model_config"],
        "model_family": family,
        "seed": seed,
        "iteration": best_step,
        "validation_score": best_score,
        "provenance": contract,
    }
    atomic_torch_save(selected, output / "best.pt")
    write_json(
        completion,
        {
            "contract": contract,
            "iterations": protocol["iterations"],
            "best_step": best_step,
            "best_validation_score": best_score,
            "best_sha256": sha256_file(output / "best.pt"),
            "completed_at_unix": time.time(),
        },
    )
    return str(output)


def eval_jobs(protocol: dict) -> list[tuple[str, int, int]]:
    return [
        (family, seed, n)
        for family in protocol["families"]
        for seed in protocol["training_seeds"]
        for n in protocol["evaluation_sizes"]
        if not (family == "mlp" and n > 24)
    ]


def eval_cell(job: tuple[str, int, int]) -> str:
    family, training_seed, n = job
    checkpoint = ART / "matched_training" / f"{family}-seed-{training_seed}/best.pt"
    path = evaluate_cell(
        output=ART / "matched_evaluation",
        method=family,
        seed=301 + training_seed,
        n=n,
        episodes=json.loads(PROTOCOL.read_text())["evaluation_episodes_per_cell"],
        checkpoint=checkpoint,
        batch_size=32,
        diagnostic_episodes=0,
    )
    return str(path.relative_to(ROOT))


def verify(protocol: dict) -> dict:
    for family in protocol["families"]:
        for seed in protocol["training_seeds"]:
            folder = ART / "matched_training" / f"{family}-seed-{seed}"
            completed = json.loads((folder / "completion.json").read_text())
            if completed["contract"] != _contract(family, seed):
                raise ValueError("matched-training contract mismatch")
            if sha256_file(folder / "best.pt") != completed["best_sha256"]:
                raise ValueError("matched-training checkpoint hash mismatch")
            records = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
            if len(records) != protocol["iterations"]:
                raise ValueError("matched-training iteration count mismatch")
            for field, expected in [
                ("training_rollouts", protocol["iterations"] * protocol["rollouts_per_iteration"]),
                ("optimizer_updates", protocol["iterations"] * protocol["updates_per_iteration"]),
                (
                    "validation_episodes",
                    protocol["iterations"]
                    // protocol["validation_interval"]
                    * protocol["validation_episodes"],
                ),
                (
                    "elite_prefix_pairs",
                    protocol["iterations"]
                    * protocol["elite_trajectories_per_iteration"]
                    * protocol["elite_prefix_length"],
                ),
            ]:
                if sum(r[field] for r in records) != expected:
                    raise ValueError(f"matched-training {field} differs from protocol")
    jobs = eval_jobs(protocol)
    for family, seed, n in jobs:
        manifest = (
            ART / "matched_evaluation" / family / f"seed-{301 + seed}" / f"n-{n}/manifest.json"
        )
        if not manifest.exists():
            raise ValueError(f"missing matched-evaluation cell: {manifest}")
        data = json.loads(manifest.read_text())
        if data["contract"]["checkpoint_sha256"] != sha256_file(
            ART / "matched_training" / f"{family}-seed-{seed}/best.pt"
        ):
            raise ValueError("matched-evaluation checkpoint mismatch")
    result = verify_bundle(ART / "matched_evaluation")
    if (
        result["cells"] != len(jobs)
        or result["graphs"] != len(jobs) * protocol["evaluation_episodes_per_cell"]
    ):
        raise ValueError("incomplete matched-evaluation panel")
    collect(ART / "matched_evaluation", ART / "matched_evaluation/evaluation.csv")
    write_json(STUDY / "matched_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "verify"])
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.stage == "verify":
        print(verify(protocol))
        return
    jobs = (
        [(f, s) for s in protocol["training_seeds"] for f in protocol["families"]]
        if args.stage == "train"
        else eval_jobs(protocol)
    )
    function = train_cell if args.stage == "train" else eval_cell
    append_jsonl(
        STUDY / "ledger.jsonl",
        {
            "event": "matched_launch",
            "stage": args.stage,
            "jobs": len(jobs),
            "protocol_sha256": sha256_file(PROTOCOL),
            "time": time.time(),
        },
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        for index, value in enumerate(pool.map(function, jobs), 1):
            append_jsonl(
                STUDY / "ledger.jsonl",
                {
                    "event": "matched_complete",
                    "stage": args.stage,
                    "cell": value,
                    "time": time.time(),
                },
            )
            print(f"matched {args.stage} {index}/{len(jobs)}", flush=True)
    if args.stage == "evaluate":
        print(verify(protocol))


if __name__ == "__main__":
    main()

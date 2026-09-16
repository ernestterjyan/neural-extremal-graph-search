"""Run the frozen corrected study, retaining resumable cells and a durable ledger."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from extremal_graph.utils import append_jsonl, sha256_file, write_json  # noqa: E402


def training_job(job):
    from extremal_graph.training import train

    family, seed = job
    config = ROOT / f"experiments/configs/corrected-{family}.toml"
    run = ROOT / f"study/artifacts/training/corrected-{family}-seed-{seed}"
    if (run / "completion.json").exists():
        from extremal_graph.config import load_training_config

        completion = json.loads((run / "completion.json").read_text())
        metadata = json.loads((run / "metadata.json").read_text())
        resolved = load_training_config(config, seed).as_dict()
        if json.loads(json.dumps(resolved)) != metadata["configuration"]:
            raise ValueError("completed training configuration differs from requested study")
        if sha256_file(run / "best.pt") != completion["checkpoint_sha256"]:
            raise ValueError("completed training checkpoint checksum mismatch")
        return {
            "family": family,
            "seed": seed,
            "status": "already_complete",
            "sha256": completion["checkpoint_sha256"],
        }
    latest = run / "latest.pt"
    checkpoint = train(config, seed_override=seed, resume=latest if latest.exists() else None)
    return {
        "family": family,
        "seed": seed,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "sha256": sha256_file(checkpoint),
    }


def evaluation_job(job):
    from extremal_graph.study import evaluate_cell

    return str(evaluate_cell(**job).relative_to(ROOT))


def expected_jobs(protocol, budget=False):
    jobs = []
    for method in protocol["families"] if budget else protocol["methods"]:
        for seed in protocol["seeds"]:
            for n in protocol["budget_sizes"] if budget else protocol["sizes"]:
                if method in {"mlp", "untrained_mlp"} and n > 24:
                    continue
                checkpoint = None
                if method in protocol["families"]:
                    filename = "budget-0050.pt" if budget else "best.pt"
                    checkpoint = str(
                        ROOT / f"study/artifacts/training/corrected-{method}-seed-{seed}" / filename
                    )
                jobs.append(
                    dict(
                        output=str(
                            ROOT
                            / ("study/artifacts/budget" if budget else "study/artifacts/evaluation")
                        ),
                        method=method,
                        seed=seed,
                        n=n,
                        episodes=protocol["episodes_per_cell"],
                        checkpoint=checkpoint,
                        batch_size=protocol["batch_size"],
                        sampling_protocol=protocol["sampling_protocol"],
                        diagnostic_episodes=0 if budget else protocol["diagnostic_episodes"],
                        diagnostic_sizes=tuple(protocol["diagnostic_sizes"]),
                    )
                )
    return jobs


def run_jobs(function, jobs, workers, stage):
    ledger = ROOT / "study/ledger.jsonl"
    append_jsonl(
        ledger,
        {
            "event": "stage_launch",
            "stage": stage,
            "jobs": len(jobs),
            "time": time.time(),
            "protocol_sha256": sha256_file(ROOT / "study/protocol.json"),
            "workers": workers,
        },
    )
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                record = {
                    "event": "completed",
                    "stage": stage,
                    "job": futures[future],
                    "result": result,
                }
            except Exception as error:
                record = {
                    "event": "failed",
                    "stage": stage,
                    "job": futures[future],
                    "error": repr(error),
                }
                failures.append(record)
            append_jsonl(ledger, record | {"time": time.time()})
            print(f"{stage}: {i}/{len(jobs)} {record['event']}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} {stage} jobs failed; see study/ledger.jsonl")


def verify_all(protocol):
    from extremal_graph.study import collect, verify_bundle

    result = {}
    for budget in [False, True]:
        name = "budget" if budget else "evaluation"
        jobs = expected_jobs(protocol, budget)
        for job in jobs:
            manifest = (
                Path(job["output"])
                / job["method"]
                / f"seed-{job['seed']}"
                / f"n-{job['n']}"
                / "manifest.json"
            )
            if not manifest.exists():
                raise ValueError(f"missing planned cell {manifest}")
            contract = json.loads(manifest.read_text())["contract"]
            for field in [
                "method",
                "seed",
                "n",
                "episodes",
                "batch_size",
                "sampling_protocol",
                "diagnostic_episodes",
            ]:
                if contract[field] != job[field]:
                    raise ValueError(f"planned cell contract mismatch: {manifest}: {field}")
            expected_hash = sha256_file(job["checkpoint"]) if job["checkpoint"] else None
            if contract["checkpoint_sha256"] != expected_hash:
                raise ValueError(f"planned checkpoint mismatch: {manifest}")
        root = ROOT / f"study/artifacts/{name}"
        result[name] = verify_bundle(root)
        if (
            result[name]["cells"] != len(jobs)
            or result[name]["graphs"] != len(jobs) * protocol["episodes_per_cell"]
        ):
            raise ValueError("study matrix mismatch")
        collect(root, ROOT / f"study/artifacts/{name}.csv")
    write_json(ROOT / "study/verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "train", "evaluate", "budget", "verify"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "study/protocol.json").read_text())
    workers = args.workers or protocol["max_workers"]
    if args.stage in {"all", "train"}:
        run_jobs(
            training_job,
            [(f, s) for s in protocol["seeds"] for f in protocol["families"]],
            workers,
            "train",
        )
    if args.stage in {"all", "evaluate"}:
        run_jobs(evaluation_job, expected_jobs(protocol), workers, "evaluate")
    if args.stage in {"all", "budget"}:
        run_jobs(evaluation_job, expected_jobs(protocol, True), workers, "budget")
    if args.stage in {"all", "verify"}:
        verify_all(protocol)


if __name__ == "__main__":
    main()

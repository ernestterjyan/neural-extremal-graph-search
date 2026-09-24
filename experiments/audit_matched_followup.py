"""Independently diagnose and replay selected matched-study trajectories."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_study_evidence import independent_prefix_diagnostics  # noqa: E402
from matched_exposure import ART, STUDY, eval_jobs  # noqa: E402

from extremal_graph.rollouts import run_episode_batch  # noqa: E402
from extremal_graph.serialization import graph_to_dict  # noqa: E402
from extremal_graph.study import digest  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.utils import write_json  # noqa: E402


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((STUDY / "matched_protocol.json").read_text())
    annotated = replays = singletons = 0
    for family, training_seed, n in eval_jobs(protocol):
        cell = ART / "matched_evaluation" / family / f"seed-{301 + training_seed}" / f"n-{n}"
        with gzip.open(cell / "chunk-000000.json.gz", "rt") as handle:
            records = json.load(handle)["records"][:2]
        if len(records) != 2:
            raise ValueError(f"missing matched-study audit records: {cell}")
        first = records[0]
        diagnostic = independent_prefix_diagnostics(n, first["actions"])
        if any(first["row"][key] != value for key, value in diagnostic.items()):
            raise ValueError(f"independent diagnosis mismatch: {first['row']['record_id']}")
        annotated += 1
        if n == 14:
            continue
        model, _ = load_model_checkpoint(
            ART / "matched_training" / f"{family}-seed-{training_seed}/best.pt"
        )
        seeds = [record["row"]["episode_seed"] for record in records]
        expected = {record["row"]["episode_seed"]: record for record in records}
        for episode_seed, trajectory in zip(
            reversed(seeds), run_episode_batch(model, n=n, seeds=list(reversed(seeds))), strict=True
        ):
            record = expected[episode_seed]
            if digest(graph_to_dict(trajectory.final_state)) != record["row"]["graph_sha256"]:
                raise ValueError("matched checkpoint/seed graph replay mismatch")
            if list(trajectory.actions) != list(map(tuple, record["actions"])):
                raise ValueError("matched checkpoint/seed action replay mismatch")
            replays += 1
        singleton = run_episode_batch(model, n=n, seeds=seeds[:1])[0]
        if list(singleton.actions) != list(map(tuple, first["actions"])):
            raise ValueError("matched singleton replay mismatch")
        singletons += 1
    result = {
        "independent_prefix_annotations": annotated,
        "reversed_pair_replays": replays,
        "singleton_replays": singletons,
        "selection": "episode 0 in every evaluation cell; reversed pair and singleton at n24/40",
        "valid": True,
    }
    write_json(STUDY / "matched_independent_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()

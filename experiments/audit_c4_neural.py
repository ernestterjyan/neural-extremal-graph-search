"""Replay saved C4 neural outputs with different batch groupings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.c4_neural_pilot import ART, STUDY, generate, model_for  # noqa: E402
from experiments.c4_solver_pilot import edges, verify_graph  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    records = [json.loads(line) for line in (STUDY / "results.jsonl").read_text().splitlines()]
    pairs = singletons = 0
    for family in protocol["families"]:
        for training_seed in protocol["training_seeds"]:
            payload = torch.load(
                ART / "training" / f"{family}-seed-{training_seed}/best.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = model_for(family)
            model.load_state_dict(payload["model_state"])
            for n in protocol["evaluation_sizes"]:
                selected = sorted(
                    (
                        record
                        for record in records
                        if record["family"] == family
                        and record["training_seed"] == training_seed
                        and record["n"] == n
                        and record["episode"] in {0, 1}
                    ),
                    key=lambda row: row["episode"],
                )
                if len(selected) != 2:
                    raise ValueError("missing neural audit episodes")
                seeds = [record["episode_seed"] for record in selected]
                expected = {record["episode_seed"]: record["edges"] for record in selected}
                reversed_runs = generate(model, n, list(reversed(seeds)))
                for seed, result in zip(reversed(seeds), reversed_runs, strict=True):
                    output = edges(result["adjacency"])
                    verify_graph(n, output)
                    if output != list(map(tuple, expected[seed])):
                        raise ValueError("C4 neural reversed-pair replay mismatch")
                    pairs += 1
                singleton = generate(model, n, seeds[:1])[0]
                if edges(singleton["adjacency"]) != list(map(tuple, selected[0]["edges"])):
                    raise ValueError("C4 neural singleton replay mismatch")
                singletons += 1
    result = {
        "valid": True,
        "reversed_pair_replays": pairs,
        "singleton_replays": singletons,
        "selection": "Episodes 0 and 1 in all 30 learned evaluation cells",
    }
    (STUDY / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()

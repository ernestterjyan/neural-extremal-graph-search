"""Independent annotations and differently grouped replays of saved follow-up graphs."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_study_evidence import independent_prefix_diagnostics  # noqa: E402
from parity_followup import ART, STUDY, generate_balance  # noqa: E402
from parity_intervention import generate as generate_parity  # noqa: E402

from extremal_graph.serialization import graph_to_dict  # noqa: E402
from extremal_graph.study import digest  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.utils import write_json  # noqa: E402


def replay(method, model, n, seeds):
    if method == "balance_parity":
        return generate_balance(n, seeds)
    return generate_parity(model, method, n, seeds)


def main():
    torch.set_num_threads(1)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    annotated = 0
    replays = 0
    singleton_replays = 0
    for method in protocol["methods"]:
        for seed in protocol["training_seeds"]:
            family = method.removesuffix("_parity")
            model = None
            if family in {"gnn", "endpoint", "candidate"}:
                model, _ = load_model_checkpoint(
                    ROOT / f"study/artifacts/training/corrected-{family}-seed-{seed}/best.pt"
                )
            for n in protocol["sizes"]:
                cell = ART / method / f"seed-{seed}" / f"n-{n}"
                with gzip.open(cell / "chunk-000000.json.gz", "rt") as handle:
                    records = json.load(handle)["records"][:2]
                for record in records:
                    diagnostic = independent_prefix_diagnostics(n, record["actions"])
                    if any(record["row"][key] != value for key, value in diagnostic.items()):
                        raise ValueError(
                            f"independent prefix mismatch: {record['row']['record_id']}"
                        )
                    annotated += 1
                if n not in [24, 40]:
                    continue
                seeds = [record["row"]["episode_seed"] for record in records]
                grouped = replay(method, model, n, list(reversed(seeds)))
                expected = {record["row"]["episode_seed"]: record for record in records}
                for episode_seed, trajectory in zip(reversed(seeds), grouped, strict=True):
                    row = expected[episode_seed]
                    if digest(graph_to_dict(trajectory.final_state)) != row["row"]["graph_sha256"]:
                        raise ValueError("follow-up checkpoint/seed replay mismatch")
                    if list(trajectory.actions) != list(map(tuple, row["actions"])):
                        raise ValueError("follow-up action replay mismatch")
                    replays += 1
                singleton = replay(method, model, n, seeds[:1])[0]
                if list(singleton.actions) != list(map(tuple, records[0]["actions"])):
                    raise ValueError("follow-up singleton replay mismatch")
                singleton_replays += 1
    result = {
        "independent_prefix_annotations": annotated,
        "reversed_pair_replays": replays,
        "singleton_replays": singleton_replays,
        "selection": "episodes 0 and 1, every method/seed/size; replays at n24 and n40",
        "valid": True,
    }
    write_json(STUDY / "independent_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()

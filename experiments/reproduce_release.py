"""Replay included historical weights without any ignored original runs directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from extremal_graph.study import collect, evaluate_cell, verify_bundle  # noqa: E402
from extremal_graph.utils import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, help="fresh output root for an independent replay")
    args = parser.parse_args()
    config = json.loads((ROOT / "study/release-evaluation.json").read_text())
    mode = "full" if args.full else "quick"
    output = args.output.resolve() if args.output else ROOT / config["output"] / mode
    for method, template in config["checkpoint_templates"].items():
        for seed in config["seeds"]:
            for n in config[mode + "_sizes"]:
                evaluate_cell(
                    output=output,
                    method=method,
                    seed=seed,
                    n=n,
                    episodes=config[mode + "_episodes"],
                    checkpoint=ROOT / template.format(seed=seed),
                    batch_size=config["batch_size"],
                    sampling_protocol=config["sampling_protocol"],
                )
    verification = verify_bundle(output)
    path = collect(output, output / "evaluation.csv")
    replay = pd.read_csv(path)
    original = pd.read_csv(ROOT / config["reference_csv"])
    keys = ["method", "seed", "n", "episode"]
    paired = replay.merge(
        original, on=keys, suffixes=("_replayed", "_original"), validate="one_to_one"
    )
    if (
        len(paired) != len(replay)
        or not (paired.edge_count_replayed == paired.edge_count_original).all()
    ):
        raise ValueError("historical count replay mismatch")
    verification["historical_counts_match"] = True
    write_json(ROOT / f"study/release-replay-{mode}.json", verification)
    print(verification)


if __name__ == "__main__":
    main()

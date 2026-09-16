"""Serial latency/throughput profile; run after other study workers have stopped."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.baselines import (  # noqa: E402
    LeastDegreePolicy,
    LookaheadPolicy,
    UniformRandomPolicy,
    run_baseline_episode,
)
from extremal_graph.rollouts import run_episode_batch  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.turan import construct_turan  # noqa: E402
from extremal_graph.utils import environment_metadata, write_json  # noqa: E402


def main():
    torch.set_num_threads(1)
    rows = []
    for method in [
        "gnn",
        "mlp",
        "candidate",
        "endpoint",
        "random",
        "least_degree",
        "lookahead",
        "turan_oracle",
    ]:
        for n in [24, 40]:
            if method == "mlp" and n > 24:
                continue
            model = None
            if method in ["gnn", "mlp", "candidate", "endpoint"]:
                model, _ = load_model_checkpoint(
                    ROOT / f"study/artifacts/training/corrected-{method}-seed-0/best.pt"
                )
            for count in [1, 8]:
                durations = []
                for repeat in range(4):
                    seeds = [9_000_000 + repeat * 100 + i for i in range(count)]
                    start = time.perf_counter()
                    if model is not None:
                        run_episode_batch(model, n=n, seeds=seeds)
                    elif method == "turan_oracle":
                        for _ in seeds:
                            construct_turan(n, 2)
                    else:
                        policy = {
                            "random": UniformRandomPolicy,
                            "least_degree": LeastDegreePolicy,
                            "lookahead": LookaheadPolicy,
                        }[method]
                        for s in seeds:
                            run_baseline_episode(n, policy(), seed=s)
                    elapsed = time.perf_counter() - start
                    if repeat:
                        durations.append(elapsed)
                rows.append(
                    dict(
                        method=method,
                        n=n,
                        batch_size=count,
                        repeats=3,
                        seconds=durations,
                        mean_seconds_per_graph=sum(durations) / 3 / count,
                    )
                )
    result = {
        "environment": environment_metadata(torch.device("cpu")),
        "rows": rows,
        "protocol": (
            "serial CPU one thread; one warmup then three repeats per cell. "
            "Count=1 is graph latency; count=8 is amortized batch generation cost. "
            "Baselines execute sequentially. Diagnostics excluded. "
            "Timing seeds excluded from study; no quality selection."
        ),
    }
    write_json(ROOT / "study/serial_inference_profile.json", result)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

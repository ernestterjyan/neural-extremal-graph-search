"""Serial cost of the three frozen supplementary arms; no study-quality selection."""

import time

import torch
from parity_intervention import ROOT, generate

from extremal_graph.training import load_model_checkpoint
from extremal_graph.utils import environment_metadata, write_json


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, _ = load_model_checkpoint(ROOT / "study/artifacts/training/corrected-gnn-seed-0/best.pt")
    rows = []
    for n in [24, 40]:
        for count in [1, 8]:
            for method in ["gnn_fresh", "gnn_parity", "uniform_parity"]:
                durations = []
                for repeat in range(4):
                    seeds = [9_100_000 + repeat * 100 + i for i in range(count)]
                    start = time.perf_counter()
                    generate(None if method == "uniform_parity" else model, method, n, seeds)
                    elapsed = time.perf_counter() - start
                    if repeat:
                        durations.append(elapsed)
                rows.append(
                    dict(
                        method=method,
                        n=n,
                        batch_size=count,
                        seconds=durations,
                        mean_seconds_per_graph=sum(durations) / 3 / count,
                    )
                )
    write_json(
        ROOT / "study/serial_intervention_profile.json",
        {
            "environment": environment_metadata(torch.device("cpu")),
            "rows": rows,
            "protocol": "Serial CPU one thread, one warmup and three repeats; seed-0 frozen GNN. "
            "Count=1 measures graph latency, count=8 amortized cost. Includes parity coloring "
            "and all construction steps; different arms can construct different edge counts. "
            "Timing seeds are excluded from quality studies; no diagnostics in timed section.",
        },
    )
    print(rows)


if __name__ == "__main__":
    main()

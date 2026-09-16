"""Regenerate six historical figures in a separate directory from retained training logs."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.reporting import generate_report  # noqa: E402
from extremal_graph.utils import sha256_file, write_json  # noqa: E402


def main():
    archive = ROOT / "study/artifacts/historical-training"
    for family in ["mvp", "mvp-mlp"]:
        for seed in range(5):
            if not (archive / f"{family}-seed-{seed}/metrics.jsonl").exists():
                raise FileNotFoundError("All ten archived training logs are required")
    output = ROOT / "study/artifacts/historical-regenerated"
    output.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="negs-historical-") as directory:
        scratch = Path(directory)
        shutil.copytree(archive, scratch / "runs")
        try:
            os.chdir(scratch)
            for name in ["mvp", "mvp-v0.2"]:
                target = scratch / "results" / name
                target.mkdir(parents=True)
                shutil.copy2(ROOT / "results" / name / "evaluation.csv", target / "evaluation.csv")
                generate_report(target / "evaluation.csv")
                figure_dir = scratch / "reports/figures"
                if name != "mvp":
                    figure_dir /= name
                for filename in [
                    "training_curve.png",
                    "optimality_ratio.png",
                    "exact_optimum_rate.png",
                ]:
                    path = figure_dir / filename
                    if not path.exists():
                        raise FileNotFoundError(f"missing required regenerated figure {path}")
                    dest = output / name / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                for filename in ["summary.csv", "summary.md"]:
                    shutil.copy2(target / filename, output / name / filename)
        finally:
            os.chdir(original)
    write_json(
        ROOT / "study/historical-regeneration.json",
        {
            "note": (
                "Six historical figures regenerated from retained inputs. "
                "Historical overlapping evaluations are not independent cohorts."
            ),
            "outputs": {
                str(p.relative_to(ROOT)): sha256_file(p) for p in output.glob("*/*") if p.is_file()
            },
        },
    )
    print("Regenerated all six historical figures without original runs directory.")


if __name__ == "__main__":
    main()

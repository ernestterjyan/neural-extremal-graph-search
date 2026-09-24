"""Recheck the C4 baseline and neural evidence from a clean local clone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from package_c4_neural import ROOT, STUDY, sha


def main() -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scratch = Path(tempfile.mkdtemp(prefix="negs-c4-check-"))
    checkout = scratch / "checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "-b",
            "research/c4-free-solver-pilot",
            str(ROOT),
            str(checkout),
        ],
        check=True,
    )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout).strip():
        raise ValueError("fresh C4 checkout is not clean")
    bundle = json.loads((STUDY / "bundle_manifest.json").read_text())
    archive = ROOT / bundle["archive"]
    if sha(archive) != bundle["sha256"]:
        raise ValueError("C4 training archive checksum mismatch")
    manifest = json.loads((ROOT / bundle["artifact_manifest"]).read_text())
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(checkout, filter="data")
    for relative, expected in manifest.items():
        if sha(checkout / relative) != expected["sha256"]:
            raise ValueError(f"C4 training artifact mismatch: {relative}")
    expected_outputs = {
        str(path.relative_to(ROOT)): sha(path)
        for pattern in [
            "study/c4_pilot/summary.csv",
            "study/c4_neural/summary.csv",
            "study/c4_neural/tables/*.csv",
            "study/c4_neural/figures/*.png",
            "study/c4_neural/report_inputs.json",
            "study/c4_neural/independent_audit.json",
        ]
        for path in ROOT.glob(pattern)
        if path.is_file()
    }
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(checkout / "src"),
        MPLCONFIGDIR=str(scratch / "matplotlib"),
        XDG_CACHE_HOME=str(scratch / "cache"),
    )
    commands = []

    def run(arguments: list[str]) -> Path:
        log = scratch / f"command-{len(commands)}.log"
        start = time.perf_counter()
        with log.open("w") as output:
            result = subprocess.run(
                [sys.executable, *arguments],
                cwd=checkout,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        commands.append(
            {
                "arguments": arguments,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - start,
                "log": str((STUDY / "validation_logs" / log.name).relative_to(ROOT)),
            }
        )
        if result.returncode:
            raise RuntimeError(f"clean C4 command failed; inspect {log}")
        print(f"passed: {' '.join(arguments[:2])}", flush=True)
        return log

    imported = (
        run(["-c", "from experiments import c4_neural_pilot; print(c4_neural_pilot.__file__)"])
        .read_text()
        .strip()
    )
    if not Path(imported).resolve().is_relative_to((checkout / "experiments").resolve()):
        raise ValueError("C4 clean validation imported source outside clone")
    run(["experiments/c4_solver_pilot.py", "verify"])
    run(["experiments/c4_solver_pilot.py", "report"])
    run(["experiments/package_c4_neural.py", "check"])
    run(["experiments/c4_neural_pilot.py", "verify"])
    run(["experiments/c4_neural_pilot.py", "report"])
    run(["experiments/analyze_c4_neural.py"])
    run(["experiments/audit_c4_neural.py"])
    run(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
    matches = {
        relative: sha(checkout / relative) == checksum
        for relative, checksum in expected_outputs.items()
    }
    if not all(matches.values()):
        raise ValueError(
            f"C4 regenerated outputs differ: {[p for p, ok in matches.items() if not ok]}"
        )
    result = {
        "valid": True,
        "source_commit": revision,
        "source_import": imported,
        "python": sys.executable,
        "training_archive_sha256": bundle["sha256"],
        "training_files_checked": len(manifest),
        "baseline_graphs": sum(1 for _ in (checkout / "study/c4_pilot/results.jsonl").open()),
        "neural_graphs": sum(1 for _ in (checkout / "study/c4_neural/results.jsonl").open()),
        "regenerated_output_matches": matches,
        "commands": commands,
        "scope": (
            "Clean source clone using the same installed Python environment and retained "
            "checkpoints; this does not independently retrain the six neural policies or "
            "test another machine."
        ),
    }
    logs = STUDY / "validation_logs"
    logs.mkdir(exist_ok=True)
    for path in scratch.glob("command-*.log"):
        shutil.copy2(path, logs / path.name)
    (STUDY / "clean_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid": True,
                "outputs_matched": len(matches),
                "graphs": result["baseline_graphs"] + result["neural_graphs"],
            }
        )
    )


if __name__ == "__main__":
    main()

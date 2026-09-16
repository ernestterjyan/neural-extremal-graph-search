"""Exercise a committed clean source checkout plus the portable local evidence archive."""

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

from package_evidence import ROOT, sha


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scratch = Path(tempfile.mkdtemp(prefix="negs-clean-validation-"))
    checkout = scratch / "checkout"
    subprocess.run(["git", "clone", "--no-hardlinks", str(ROOT), str(checkout)], check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout).strip():
        raise ValueError("initial validation checkout is not clean")
    bundle = json.loads((ROOT / "study/bundle-manifest.json").read_text())
    archive = ROOT / bundle["archive"]
    if sha(archive) != bundle["sha256"]:
        raise ValueError("archive checksum mismatch")
    files = json.loads((ROOT / bundle["file_manifest"]).read_text())
    with tarfile.open(archive) as handle:
        handle.extractall(checkout, filter="data")
    for relative, expected in files.items():
        if sha(checkout / relative) != expected["sha256"]:
            raise ValueError(f"extracted evidence mismatch: {relative}")
    expected_outputs = {
        str(p.relative_to(ROOT)): sha(p)
        for pattern in [
            "study/tables/*.csv",
            "study/figures/*.png",
            "study/artifacts/historical-regenerated/*/*",
        ]
        for p in ROOT.glob(pattern)
        if p.is_file()
    }
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(checkout / "src"),
        MPLCONFIGDIR=str(scratch / "matplotlib"),
        XDG_CACHE_HOME=str(scratch / "cache"),
    )
    commands = []

    def run(arguments):
        number = len(commands)
        log = scratch / f"command-{number}.log"
        begun = time.perf_counter()
        with log.open("w") as handle:
            completed = subprocess.run(
                [sys.executable, *arguments],
                cwd=checkout,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        commands.append(
            {
                "arguments": arguments,
                "exit_code": completed.returncode,
                "seconds": time.perf_counter() - begun,
                "log": str(log),
            }
        )
        if completed.returncode:
            raise RuntimeError(f"clean-checkout command failed; inspect {log}")
        print(f"passed: {' '.join(arguments[:2])}", flush=True)
        return log

    import_log = run(["-c", "import extremal_graph; print(extremal_graph.__file__)"])
    imported = import_log.read_text().strip()
    if not imported.startswith(str(checkout / "src")):
        raise ValueError("validation imported source outside clean checkout")
    run(["experiments/run_study.py", "verify"])
    run(["experiments/analyze_study.py"])
    run(["experiments/parity_intervention.py"])
    run(["experiments/analyze_intervention.py"])
    run(["experiments/regenerate_historical_reports.py"])
    matches = {p: sha(checkout / p) == checksum for p, checksum in expected_outputs.items()}
    if not all(matches.values()):
        raise ValueError(
            f"regenerated outputs differ: {[p for p, ok in matches.items() if not ok]}"
        )
    validation_relative = f"study/artifacts/clean-checkout-validations/{scratch.name}"
    replay_relative = validation_relative + "/release-replay"
    if (checkout / replay_relative).exists():
        raise ValueError("fresh replay destination already exists in evidence archive")
    run(["experiments/reproduce_release.py", "--output", replay_relative])
    replay = json.loads((checkout / "study/release-replay-quick.json").read_text())
    if replay["graphs"] != 640 or not replay["historical_counts_match"]:
        raise ValueError("fresh included-weight replay failed")
    result = {
        "valid": True,
        "source_commit": revision,
        "clean_checkout": str(checkout),
        "source_import": imported,
        "python": sys.executable,
        "scope": "Fresh clone and checksummed archive extraction. Shared preinstalled dependency "
        "environment; not a new-machine install or full independent retraining.",
        "input_archive_sha256": bundle["sha256"],
        "extracted_files_checked": len(files),
        "regenerated_output_matches": matches,
        "commands": commands,
        "main_and_budget_verification": json.loads(
            (checkout / "study/verification.json").read_text()
        ),
        "supplementary_verification": json.loads(
            (checkout / "study/intervention-verification.json").read_text()
        ),
        "fresh_release_replay": replay,
        "note": "Fresh replay artifacts and validation command logs are copied into the handoff "
        "after checking; the final archive adds these validation outputs to the checked inputs.",
    }
    destination = ROOT / replay_relative
    if destination.exists():
        raise ValueError("refusing to overwrite prior fresh-replay evidence")
    shutil.copytree(checkout / replay_relative, destination)
    logs = ROOT / validation_relative / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for log in scratch.glob("command-*.log"):
        shutil.copy2(log, logs / log.name)
    (ROOT / validation_relative / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "study/clean_checkout_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid": True, "outputs_matched": len(matches), "fresh_replay_graphs": 640}))


if __name__ == "__main__":
    main()

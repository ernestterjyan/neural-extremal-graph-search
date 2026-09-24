"""Validate the committed follow-up source and evidence in a fresh local clone."""

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

from package_followup import ROOT, STUDY, sha


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scratch = Path(tempfile.mkdtemp(prefix="negs-followup-check-"))
    checkout = scratch / "checkout"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "-b",
            "research/parity-mechanism-followup",
            str(ROOT),
            str(checkout),
        ],
        check=True,
    )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout).strip():
        raise ValueError("fresh follow-up source checkout is not clean")
    bundle = json.loads((STUDY / "bundle_manifest.json").read_text())
    archive = ROOT / bundle["archive"]
    if sha(archive) != bundle["sha256"]:
        raise ValueError("follow-up evidence checksum mismatch")
    files = json.loads((ROOT / bundle["artifact_manifest"]).read_text())
    with tarfile.open(archive) as handle:
        handle.extractall(checkout, filter="data")
    for relative, expected in files.items():
        if sha(checkout / relative) != expected["sha256"]:
            raise ValueError(f"extracted follow-up evidence mismatch: {relative}")
    expected_outputs = {
        str(path.relative_to(ROOT)): sha(path)
        for pattern in [
            "study/followup/tables/*.csv",
            "study/followup/figures/*.png",
            "study/followup/lookahead_reachability.json",
            "study/followup/independent_audit.json",
            "study/followup/matched_independent_audit.json",
            "study/followup/verification.json",
            "study/followup/matched_verification.json",
            "study/followup/parity_report_inputs.json",
            "study/followup/matched_report_inputs.json",
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

    def run(arguments):
        log = scratch / f"command-{len(commands)}.log"
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
            raise RuntimeError(f"clean follow-up command failed; inspect {log}")
        print(f"passed: {' '.join(arguments[:2])}", flush=True)
        return log

    imported = (
        run(["-c", "import extremal_graph; print(extremal_graph.__file__)"]).read_text().strip()
    )
    if not imported.startswith(str(checkout / "src")):
        raise ValueError("follow-up validation imported source outside clone")
    run(["experiments/parity_followup.py", "verify"])
    run(["experiments/matched_exposure.py", "verify"])
    run(["experiments/analyze_parity_followup.py"])
    run(["experiments/analyze_matched_exposure.py"])
    run(["experiments/audit_followup.py"])
    run(["experiments/audit_matched_followup.py"])
    run(["experiments/lookahead_reachability.py"])
    run(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
    matches = {
        relative: sha(checkout / relative) == checksum
        for relative, checksum in expected_outputs.items()
    }
    if not all(matches.values()):
        raise ValueError(
            f"regenerated follow-up outputs differ: {[p for p, ok in matches.items() if not ok]}"
        )
    result = {
        "valid": True,
        "source_commit": revision,
        "source_import": imported,
        "python": sys.executable,
        "input_archive_sha256": bundle["sha256"],
        "extracted_files_checked": len(files),
        "regenerated_output_matches": matches,
        "commands": commands,
        "parity_graphs": json.loads((checkout / "study/followup/verification.json").read_text())[
            "graphs"
        ],
        "matched_graphs": json.loads(
            (checkout / "study/followup/matched_verification.json").read_text()
        )["graphs"],
        "scope": (
            "Clean source clone and checksummed evidence extraction using the same "
            "preinstalled Python environment. This does not independently retrain "
            "twenty models or test a new machine."
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
                "graphs": result["parity_graphs"] + result["matched_graphs"],
            }
        )
    )


if __name__ == "__main__":
    main()

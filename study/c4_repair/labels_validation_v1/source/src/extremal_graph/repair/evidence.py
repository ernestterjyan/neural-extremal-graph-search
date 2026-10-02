"""Immutable run contracts, source snapshots, and append-only experiment records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_SIZES = (20, 24, 28, 32, 36, 40)
VALIDATION_SIZES = (22, 26, 30, 34, 38)
EVALUATION_SIZES = (44, 48, 52, 56, 64, 72, 80, 96)
REPRODUCTION_SIZES = {"validation": (22, 30, 38), "evaluation": (44, 64, 96)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment() -> dict:
    cpu_model = platform.processor()
    verified_model = False
    if platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            verified_model = True
        except (OSError, subprocess.CalledProcessError):
            pass  # Some sandboxes allow computation but block hardware identification.
    elif Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        cpu_model = next(
            (line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")),
            cpu_model,
        )
        verified_model = bool(cpu_model)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "cpu_workers": 1,
        "hardware": {
            "cpu_model": cpu_model,
            "logical_cpus": os.cpu_count(),
            "architecture": platform.machine(),
            "verified_cpu_model": verified_model,
        },
        "pid": os.getpid(),
        "packages": {
            p: importlib.metadata.version(p) for p in ["ortools", "torch", "numpy", "networkx"]
        },
    }


def source_manifest() -> dict:
    files = sorted((ROOT / "src/extremal_graph").rglob("*.py"))
    files += sorted((ROOT / "experiments").glob("c4_repair*.py"))
    files += [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "README.md",
        ROOT / "LICENSE",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in files if path.exists()}


def independent_audit(directory: Path, *, allow_partial: bool = False) -> dict:
    """Run the separate standard-library checker before using scientific evidence."""
    path = ROOT / "experiments/c4_repair_audit.py"
    spec = importlib.util.spec_from_file_location("campaign_independent_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit(directory, allow_partial=allow_partial)


def require_input_replay(directory: Path) -> None:
    report = json.loads((directory / "INPUT_REPLAY.json").read_text())
    contract = verify_contract(directory)
    if (
        not report["complete_panel"]
        or not report["all_completed_inputs_replayed"]
        or report["results_sha256"] != sha(directory / "results.jsonl")
        or report["source_sha256"] != contract["source_sha256"]
        or report["protocol_sha256"] != contract["protocol_sha256"]
        or report["replay_program_sha256"] != sha(directory / "INPUT_REPLAY_replay.py")
        or report["independent_auditor_sha256"] != sha(directory / "INPUT_REPLAY_auditor.py")
    ):
        raise ValueError("complete retained deterministic input replay required")
    records = load_records(directory / "results.jsonl")

    def identity(row):
        return (
            row["n"],
            row.get("replication", 0),
            row.get("block", 0),
            row.get("trial", row.get("method", "repair")),
        )

    cells = report["cells"]
    if len(cells) != len(records) or {identity(c) for c in cells} != {identity(r) for r in records}:
        raise ValueError("input replay omitted or duplicated experimental cells")
    actual = {identity(r): r for r in records}
    if any(
        c["input_replayed"] is not (actual[identity(c)].get("status", "complete") == "complete")
        for c in cells
    ):
        raise ValueError("input replay completion differs from retained evidence")


def prepare_batch(destination: Path, protocol: dict) -> dict:
    if destination.exists():
        raise FileExistsError("evidence directory already exists; use a new batch name")
    destination.mkdir(parents=True)
    sources = source_manifest()
    for relative in sources:
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    contract = {
        "protocol": protocol,
        "source": sources,
        "source_sha256": json_sha(sources),
        "protocol_sha256": json_sha(protocol),
        "environment": environment(),
        "git_commit": commit,
    }
    (destination / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return contract


def prepare_or_resume(destination: Path, protocol: dict) -> dict:
    """Resume only an unchanged batch on the same runtime/hardware."""
    if not destination.exists():
        return prepare_batch(destination, protocol)
    contract = verify_contract(destination)
    if (
        json_sha(protocol) != contract["protocol_sha256"]
        or json_sha(source_manifest()) != contract["source_sha256"]
    ):
        raise ValueError("cannot resume a batch with changed methods, source, or protocol")
    current = environment()
    for key in [
        "platform",
        "machine",
        "processor",
        "python",
        "cpu_workers",
        "packages",
        "hardware",
    ]:
        if current[key] != contract["environment"][key]:
            raise ValueError(f"cannot resume after environment change: {key}")
    return contract


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_contract(directory: Path) -> dict:
    contract = json.loads((directory / "contract.json").read_text())
    if json_sha(contract["protocol"]) != contract["protocol_sha256"]:
        raise ValueError("protocol digest mismatch")
    if json_sha(contract["source"]) != contract["source_sha256"]:
        raise ValueError("source manifest digest mismatch")
    for relative, digest in contract["source"].items():
        if sha(directory / "source" / relative) != digest:
            raise ValueError(f"source snapshot mismatch: {relative}")
    return contract


def require_development(sizes: list[int]) -> None:
    if not sizes or any(not 2 <= n <= 40 for n in sizes):
        raise ValueError(
            "development runs must use orders 2..40; held-out evaluation requires a freeze"
        )

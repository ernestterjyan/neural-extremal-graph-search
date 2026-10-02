"""Immutable run contracts, source snapshots, and append-only experiment records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_SIZES = (20, 24, 28, 32, 36, 40)
VALIDATION_SIZES = (22, 26, 30, 34, 38)
EVALUATION_SIZES = (44, 48, 52, 56, 64, 72, 80, 96)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_workers": 1,
        "pid": os.getpid(),
        "packages": {
            p: importlib.metadata.version(p) for p in ["ortools", "torch", "numpy", "networkx"]
        },
    }


def source_manifest() -> dict:
    files = sorted((ROOT / "src/extremal_graph/repair").glob("*.py"))
    files += [ROOT / "experiments/c4_repair.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {str(path.relative_to(ROOT)): sha(path) for path in files if path.exists()}


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

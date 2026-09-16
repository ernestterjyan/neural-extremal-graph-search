"""Reproducibility and artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "mps"}:
        raise ValueError("device must be 'cpu', 'mps', or 'auto'")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def code_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def environment_metadata(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "numpy", "networkx", "pandas", "matplotlib", "seaborn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "code_commit": code_commit(),
        "source": source_manifest(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "packages": packages,
        "process_id": os.getpid(),
    }


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("experiments/configs/*.toml"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    )
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in files if p.is_file()}
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    return {
        "root": str(root),
        "files": hashes,
        "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "git_status": status.stdout.strip() if status.returncode == 0 else None,
    }


def save_source_snapshot(path: Path) -> dict[str, Any]:
    manifest = source_manifest()
    root = Path(manifest["root"])
    with tarfile.open(path, "w:gz") as archive:
        for name in manifest["files"]:
            archive.add(root / name, arcname=name)
    return {"path": path.name, "sha256": sha256_file(path), "source_sha256": manifest["sha256"]}


def capture_random_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        value["mps"] = torch.mps.get_rng_state()
    return value


def restore_random_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if "mps" in value and torch.backends.mps.is_available():
        torch.mps.set_rng_state(value["mps"].cpu())


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

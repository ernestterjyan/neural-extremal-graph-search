"""Create and check a portable archive of the verified follow-up evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study/followup"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package() -> dict:
    parity = json.loads((STUDY / "verification.json").read_text())
    matched = json.loads((STUDY / "matched_verification.json").read_text())
    if parity != {"cells": 100, "graphs": 10000, "valid": True}:
        raise ValueError("follow-up parity evidence is not complete and verified")
    if matched != {"cells": 55, "graphs": 5500, "valid": True}:
        raise ValueError("follow-up matched evidence is not complete and verified")
    files = sorted(path for path in (STUDY / "artifacts").rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("evidence must consist of ordinary files")
    manifest = {
        str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in files
    }
    manifest_path = STUDY / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    archive = STUDY / "evidence-v1.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for path in files:
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as content:
                tar.addfile(info, content)
    result = {
        "archive": str(archive.relative_to(ROOT)),
        "sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "files": len(files),
        "uncompressed_bytes": sum(item["bytes"] for item in manifest.values()),
        "artifact_manifest": str(manifest_path.relative_to(ROOT)),
        "artifact_manifest_sha256": sha(manifest_path),
        "parity_protocol_sha256": sha(STUDY / "protocol.json"),
        "matched_protocol_sha256": sha(STUDY / "matched_protocol.json"),
        "parity_graphs": parity["graphs"],
        "matched_graphs": matched["graphs"],
        "note": (
            "Includes graph chunks, trajectories, source snapshots, complete training "
            "records, checkpoints, and both CSV evaluations. The source checkout and "
            "tracked analysis reports accompany this archive."
        ),
    }
    (STUDY / "bundle_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def check() -> dict:
    bundle = json.loads((STUDY / "bundle_manifest.json").read_text())
    archive = ROOT / bundle["archive"]
    if sha(archive) != bundle["sha256"]:
        raise ValueError("follow-up archive hash mismatch")
    if sha(ROOT / bundle["artifact_manifest"]) != bundle["artifact_manifest_sha256"]:
        raise ValueError("follow-up manifest hash mismatch")
    files = json.loads((ROOT / bundle["artifact_manifest"]).read_text())
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or member.name not in files:
                raise ValueError("unexpected archive member")
            if (
                hashlib.sha256(handle.extractfile(member).read()).hexdigest()
                != files[member.name]["sha256"]
            ):
                raise ValueError(f"archive member differs: {member.name}")
            if member.size != files[member.name]["bytes"]:
                raise ValueError(f"archive member size differs: {member.name}")
            seen.add(member.name)
    if seen != set(files):
        raise ValueError("missing archive members")
    return {"valid": True, "files_checked": len(seen), "archive_sha256": bundle["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["package", "check"])
    args = parser.parse_args()
    print(json.dumps(package() if args.stage == "package" else check()))


if __name__ == "__main__":
    main()

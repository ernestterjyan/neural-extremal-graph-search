"""Build a deterministic local evidence archive and per-file checksum manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    files = sorted(p for p in (ROOT / "study/artifacts").rglob("*") if p.is_file())
    if not files:
        raise ValueError("no evidence to package")
    if any(p.is_symlink() for p in files):
        raise ValueError("evidence package must contain ordinary files")
    manifest = {
        str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files
    }
    manifest_path = ROOT / "study/artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    archive = ROOT / "study/evidence-corrected-v1.tar.gz"
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
        "file_count": len(files),
        "uncompressed_bytes": sum(x["bytes"] for x in manifest.values()),
        "file_manifest": str(manifest_path.relative_to(ROOT)),
        "file_manifest_sha256": sha(manifest_path),
        "scope": "Local evidence archive. Extract at repository root with matching source "
        "checkout. "
        "Primary and supplementary research cohorts, historical inputs/replays, and engineering "
        "checks are distinct namespaces; do not count replay copies as new observations.",
    }
    (ROOT / "study/bundle-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

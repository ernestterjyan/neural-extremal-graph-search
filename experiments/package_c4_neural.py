"""Package the C4 neural pilot's complete training evidence for clean replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study/c4_neural"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package() -> dict:
    if not (STUDY / "results.jsonl").exists():
        raise ValueError("C4 neural evaluation is not complete")
    files = sorted(path for path in (STUDY / "artifacts").rglob("*") if path.is_file())
    if len(files) != 12 or any(path.is_symlink() for path in files):
        raise ValueError("expected two files for each of six training runs")
    manifest = {
        str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in files
    }
    manifest_path = STUDY / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    archive = STUDY / "training-evidence-v1.tar.gz"
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
    bundle = {
        "archive": str(archive.relative_to(ROOT)),
        "sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "files": len(files),
        "artifact_manifest": str(manifest_path.relative_to(ROOT)),
        "artifact_manifest_sha256": sha(manifest_path),
        "protocol_sha256": sha(STUDY / "protocol.json"),
        "evaluation_sha256": sha(STUDY / "results.jsonl"),
        "scope": (
            "Six selected neural checkpoints and complete 80-iteration training logs; "
            "the separate tracked results.jsonl retains all 3,000 held-out final graphs."
        ),
    }
    (STUDY / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def check() -> dict:
    bundle = json.loads((STUDY / "bundle_manifest.json").read_text())
    if sha(ROOT / bundle["archive"]) != bundle["sha256"]:
        raise ValueError("C4 neural archive checksum mismatch")
    if sha(ROOT / bundle["artifact_manifest"]) != bundle["artifact_manifest_sha256"]:
        raise ValueError("C4 neural manifest checksum mismatch")
    manifest = json.loads((ROOT / bundle["artifact_manifest"]).read_text())
    found = set()
    with tarfile.open(ROOT / bundle["archive"], "r:gz") as handle:
        for member in handle:
            if not member.isfile() or member.name not in manifest:
                raise ValueError("unexpected C4 neural archive member")
            content = handle.extractfile(member).read()
            if (
                hashlib.sha256(content).hexdigest() != manifest[member.name]["sha256"]
                or len(content) != manifest[member.name]["bytes"]
            ):
                raise ValueError("C4 neural archive member mismatch")
            found.add(member.name)
    if found != set(manifest):
        raise ValueError("missing C4 neural archive member")
    return {"valid": True, "files": len(found), "sha256": bundle["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["package", "check"])
    args = parser.parse_args()
    print(json.dumps(package() if args.stage == "package" else check()))


if __name__ == "__main__":
    main()

"""Adaptive selection, balanced order and process exclusion for the repair campaign.

This module does not launch a solver. It supplies a family/size adaptive selector,
balanced method ordering and a POSIX process lease for sequential timed experiments.
"""

from __future__ import annotations

import errno
import json
import math
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar


class Candidate(Protocol):
    vertices: tuple[int, ...]
    family: str


CandidateT = TypeVar("CandidateT", bound=Candidate)
Arm = tuple[str, int]


def arm(region: Candidate) -> Arm:
    return region.family, len(region.vertices)


class AdaptiveRegions:
    """Exploratory reward-rate adaptation over family × actual region size.

    The engine reports the obtained gain and elapsed repair time. No optimum label
    or uncharged oracle is used. Selection overhead is part of the calling search.
    """

    def __init__(
        self, *, exploration: float = 0.15, smoothing: float = 0.2, reward_scale: float = 5.0
    ):
        if (
            not math.isfinite(exploration)
            or not 0 < exploration <= 1
            or not math.isfinite(smoothing)
            or not 0 < smoothing <= 1
            or not math.isfinite(reward_scale)
            or reward_scale <= 0
        ):
            raise ValueError("invalid adaptive selection configuration")
        self.exploration = exploration
        self.smoothing = smoothing
        self.reward_scale = reward_scale
        self.weights: dict[Arm, float] = {}
        self.observations: dict[Arm, dict] = {}

    def probabilities(self, candidates: list[Candidate]) -> dict[Arm, float]:
        available = sorted({arm(region) for region in candidates})
        if not available:
            raise ValueError("empty candidate pool")
        weights = {key: self.weights.get(key, 1.0) for key in available}
        total = sum(weights.values())
        return {
            key: self.exploration / len(available) + (1 - self.exploration) * weights[key] / total
            for key in available
        }

    def choose(self, candidates: list[CandidateT], rng: random.Random) -> CandidateT:
        probabilities = self.probabilities(candidates)
        keys = list(probabilities)
        selected = rng.choices(keys, [probabilities[key] for key in keys])[0]
        return rng.choice([region for region in candidates if arm(region) == selected])

    def observe(self, region: Candidate, *, gain: int, seconds: float) -> None:
        if type(gain) is not int or gain < 0 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid obtained repair outcome")
        key = arm(region)
        reward = 1.0 + self.reward_scale * gain / max(0.01, seconds)
        self.weights[key] = (1 - self.smoothing) * self.weights.get(
            key, 1.0
        ) + self.smoothing * reward
        observation = self.observations.setdefault(key, {"repairs": 0, "gain": 0, "seconds": 0.0})
        observation["repairs"] += 1
        observation["gain"] += gain
        observation["seconds"] += seconds

    def report(self) -> list[dict]:
        return [
            {
                "family": key[0],
                "region_size": key[1],
                "weight": self.weights[key],
                **self.observations[key],
            }
            for key in sorted(self.observations)
        ]


def balanced_method_order(
    methods: list[str], *, seed_offset: int, n: int, replication: int, block: int
) -> list[str]:
    """Seeded permutation per (size, replication), rotated across model blocks.

    With four methods and eight blocks, each method appears twice in every position
    for each paired size/replication group. No model or solver result affects order.
    """
    if not methods or len(set(methods)) != len(methods) or block < 0 or replication < 0:
        raise ValueError("invalid paired method order")
    base = sorted(methods)
    random.Random(seed_offset + n * 10000 + replication).shuffle(base)
    offset = block % len(base)
    return base[offset:] + base[:offset]


def default_lease_path() -> Path:
    # A host-wide path also excludes campaigns in different checkouts/worktrees.
    return Path(tempfile.gettempdir()) / f"negs-c4-repair-worker-{os.getuid()}.lock"


class ExperimentLease:
    """Nonblocking OS lock held by one experiment/training process on this host.

    The file is never unlinked: unlinking a held lock can create two independent
    inodes and permit concurrent experiments. The OS releases it when a process dies.
    Owner JSON is diagnostic metadata; it is not evidence that the process is live.
    """

    def __init__(self, *, stage: str, batch: Path, path: Path | None = None):
        self.path = default_lease_path() if path is None else path
        self.owner = {
            "pid": os.getpid(),
            "stage": stage,
            "batch": str(batch.resolve()),
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "running",
        }
        self.stream = None

    def _write(self) -> None:
        self.stream.seek(0)
        self.stream.truncate()
        json.dump(self.owner, self.stream, sort_keys=True)
        self.stream.write("\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __enter__(self):
        import fcntl

        if self.stream is not None:
            raise RuntimeError("experiment lease is already entered")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.seek(0)
            details = stream.read()
            stream.close()
            if error.errno not in {errno.EAGAIN, errno.EACCES}:
                raise
            raise RuntimeError(f"another C4 campaign process holds the lease: {details}") from error
        self.stream = stream
        try:
            self.owner.update(
                pid=os.getpid(), status="running", started_utc=datetime.now(UTC).isoformat()
            )
            self._write()
        except BaseException:
            stream.close()
            self.stream = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import fcntl

        try:
            self.owner.update(status="failed" if exc_type else "released")
            self._write()
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None

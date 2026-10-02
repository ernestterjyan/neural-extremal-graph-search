"""Selection fairness and real cross-process exclusion for staged campaign controls."""

import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/extremal_graph/repair/controls.py"
spec = importlib.util.spec_from_file_location("repair_controls", SCRIPT)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@dataclass(frozen=True)
class Region:
    vertices: tuple[int, ...]
    family: str


def candidates():
    return [
        Region(tuple(range(size)), family)
        for family in ["random", "neighborhood", "blocking_path"]
        for size in [3, 4, 5]
    ]


def test_adaptation_can_learn_size_within_one_family():
    selector = controls.AdaptiveRegions()
    pool = candidates()
    small = next(r for r in pool if r.family == "random" and len(r.vertices) == 3)
    large = next(r for r in pool if r.family == "random" and len(r.vertices) == 5)
    for _ in range(20):
        selector.observe(small, gain=1, seconds=0.1)
        selector.observe(large, gain=1, seconds=2)
    probabilities = selector.probabilities(pool)
    assert probabilities[("random", 3)] > probabilities[("random", 5)]
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(p >= selector.exploration / 9 for p in probabilities.values())
    assert selector.report()[0]["repairs"] == 20


def test_missing_arms_and_duplicate_draws_do_not_bias_arm_probabilities():
    selector = controls.AdaptiveRegions()
    first, second = candidates()[:2]
    pool = [first] * 8 + [second]
    assert selector.probabilities(pool) == {
        (first.family, 3): pytest.approx(0.5),
        (second.family, 4): pytest.approx(0.5),
    }
    rng1, rng2 = random.Random(17), random.Random(17)
    assert [selector.choose(pool, rng1) for _ in range(20)] == [
        selector.choose(pool, rng2) for _ in range(20)
    ]
    assert selector.choose([first], random.Random(1)) == first
    with pytest.raises(ValueError, match="empty"):
        selector.choose([], random.Random(1))


@pytest.mark.parametrize(
    "kwargs", [{"exploration": 0}, {"smoothing": 1.1}, {"reward_scale": float("nan")}]
)
def test_invalid_adaptive_configuration(kwargs):
    with pytest.raises(ValueError, match="configuration"):
        controls.AdaptiveRegions(**kwargs)


def test_balanced_order_for_every_paired_group():
    methods = ["gnn", "mlp", "adaptive", "tabu"]
    for n in [22, 26, 30, 34, 38]:
        for replication in range(10):
            orders = [
                controls.balanced_method_order(
                    methods, seed_offset=8000000, n=n, replication=replication, block=block
                )
                for block in range(8)
            ]
            assert all(set(order) == set(methods) for order in orders)
            for position in range(4):
                assert Counter(order[position] for order in orders) == dict.fromkeys(methods, 2)
    assert controls.balanced_method_order(
        methods, seed_offset=123, n=22, replication=0, block=0
    ) == controls.balanced_method_order(
        methods[::-1], seed_offset=123, n=22, replication=0, block=0
    )


def child(path, *, crash=False):
    program = """
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('controls', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    with module.ExperimentLease(
        stage='test-child', batch=pathlib.Path('.'), path=pathlib.Path(sys.argv[2])
    ):
        if sys.argv[3] == 'crash':
            os._exit(3)
except RuntimeError as error:
    print(error)
    sys.exit(2)
"""
    return subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT), str(path), "crash" if crash else "normal"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_real_process_exclusion_and_release(tmp_path):
    path = tmp_path / "worker.lock"
    with controls.ExperimentLease(stage="test-parent", batch=tmp_path, path=path):
        result = child(path)
        assert result.returncode == 2 and "test-parent" in result.stdout
        assert json.loads(path.read_text())["status"] == "running"
    assert json.loads(path.read_text())["status"] == "released"
    assert path.exists() and child(path).returncode == 0


def test_dead_process_releases_os_lock_despite_stale_owner_file(tmp_path):
    path = tmp_path / "worker.lock"
    assert child(path, crash=True).returncode == 3
    assert json.loads(path.read_text())["status"] == "running"
    assert child(path).returncode == 0

"""Timed multistart tabu and shared large-neighborhood search trajectories."""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .constructions import starting_pool
from .engine import repair
from .graph import Graph, greedy_fill, verify_witness
from .regions import Region, candidate_regions

FAMILIES = ("random", "neighborhood", "blocking_path")


@dataclass(frozen=True)
class SearchConfig:
    seconds: float = 60.0
    region_sizes: tuple[int, ...] = (3, 5, 7)
    per_family: int = 2
    repair_seconds: float = 1.0
    stagnation_repairs: int = 10
    tabu_restart_iterations: int = 400
    tabu_tenure_min: int = 7
    tabu_tenure_max: int = 15
    tabu_penalty: float = 1.0


class SearchClock:
    def __init__(self, n: int, seconds: float):
        if seconds <= 0:
            raise ValueError("positive search budget required")
        self.started = time.perf_counter()
        self.deadline = self.started + seconds
        self.budget = seconds
        self.best = Graph.empty(n)
        self.history = [{"seconds": 0.0, "edges": 0, "graph_edges": [], "origin": "empty"}]

    def offer(self, graph: Graph, origin: str, *, completed_at: float | None = None) -> None:
        timestamp = time.perf_counter() if completed_at is None else completed_at
        if timestamp <= self.deadline and graph.m > self.best.m:
            self.best = graph.copy()
            self.history.append(
                {
                    "seconds": timestamp - self.started,
                    "edges": graph.m,
                    "graph_edges": graph.edges(),
                    "origin": origin,
                }
            )

    def report(self) -> dict:
        witnesses = {}
        for checkpoint in [1.0, 10.0, 60.0]:
            if checkpoint <= self.budget:
                witnesses[str(int(checkpoint))] = max(
                    (event for event in self.history if event["seconds"] <= checkpoint),
                    key=lambda event: event["edges"],
                )
        elapsed = time.perf_counter() - self.started
        return {
            "edges": self.best.edges(),
            "edge_count": self.best.m,
            "history": self.history,
            "checkpoints": witnesses,
            "elapsed_seconds": elapsed,
            "budget_seconds": self.budget,
            "deadline_overshoot_seconds": max(0, elapsed - self.budget),
        }


def initialize(n: int, seed: int, clock: SearchClock):
    # All solvers build the same pool in the same order, inside the budget.
    pool = starting_pool(n, seed, deadline=clock.deadline)
    for start in pool:
        clock.offer(start.graph, f"start:{start.kind}", completed_at=start.completed_at)
    return pool


def tabu_search(n: int, seed: int, config: SearchConfig) -> dict:
    clock = SearchClock(n, config.seconds)
    rng = random.Random(seed)
    pool = initialize(n, seed, clock)
    initialization = time.perf_counter() - clock.started
    current = clock.best.copy()
    cycles = 0
    penalty = config.tabu_penalty
    tabu_until = {}
    iterations = stagnation = restarts = 0
    feasible_window = 0
    pairs = list(itertools.combinations(range(n), 2))
    while time.perf_counter() < clock.deadline:
        best_score = -float("inf")
        choices = []
        for index, (u, v) in enumerate(pairs):
            if index % 64 == 0 and time.perf_counter() >= clock.deadline:
                break
            present = current.has(u, v)
            delta_cycles = (
                -current.deletion_cycles(u, v) if present else current.addition_cycles(u, v)
            )
            new_cycles = cycles + delta_cycles
            new_edges = current.m + (-1 if present else 1)
            aspiration = new_cycles == 0 and new_edges > clock.best.m
            if tabu_until.get((u, v), -1) > iterations and not aspiration:
                continue
            score = new_edges - penalty * new_cycles
            if score > best_score + 1e-10:
                best_score, choices = score, [(u, v, delta_cycles)]
            elif abs(score - best_score) <= 1e-10:
                choices.append((u, v, delta_cycles))
        if time.perf_counter() >= clock.deadline:
            break
        if not choices:
            tabu_until.clear()
            continue
        u, v, delta = rng.choice(choices)
        if current.has(u, v):
            current.remove(u, v)
        else:
            current.add(u, v)
        cycles += delta
        if cycles < 0:
            raise RuntimeError("negative incremental C4 count")
        tabu_until[(u, v)] = iterations + rng.randint(
            config.tabu_tenure_min, config.tabu_tenure_max
        )
        iterations += 1
        stagnation += 1
        if cycles == 0:
            before = clock.best.m
            clock.offer(current, "tabu")
            if clock.best.m > before:
                stagnation = 0
            feasible_window += 1
        # Adjust penalty in both directions rather than trapping the trajectory
        # permanently in the feasible set or in an infeasible dense state.
        if iterations % 25 == 0:
            if feasible_window > 12:
                penalty = max(0.1, penalty / 1.2)
            elif feasible_window < 3:
                penalty = min(100, penalty * 1.2)
            feasible_window = 0
        if stagnation >= config.tabu_restart_iterations:
            current = rng.choice(pool).graph.copy() if pool else clock.best.copy()
            cycles = 0
            for edge in rng.sample(current.edges(), min(3, current.m)):
                current.remove(*edge)
            greedy_fill(current, rng, deadline=clock.deadline)
            clock.offer(current, "tabu_restart")
            tabu_until.clear()
            penalty = config.tabu_penalty
            stagnation = 0
            restarts += 1
    result = clock.report()
    verify_witness(n, result["edges"])
    result.update(
        method="tabu",
        n=n,
        seed=seed,
        initialization_seconds=initialization,
        starting_pool=[
            {
                "kind": s.kind,
                "field_order": s.field_order,
                "sha256": s.graph.digest(),
                "edges": s.graph.edges(),
            }
            for s in pool
        ],
        iterations=iterations,
        restarts=restarts,
        configuration=asdict(config),
    )
    return result


Selector = Callable[[Graph, list[Region]], list[float]]


def neighborhood_search(
    n: int,
    seed: int,
    config: SearchConfig,
    *,
    method: str = "adaptive",
    selector: Selector | None = None,
) -> dict:
    if method not in {"adaptive", "random", "mlp", "gnn"}:
        raise ValueError("unknown neighborhood method")
    if method in {"mlp", "gnn"} and selector is None:
        raise ValueError("learned method requires a selector")
    clock = SearchClock(n, config.seconds)
    rng = random.Random(seed)
    pool = initialize(n, seed, clock)
    initialization = time.perf_counter() - clock.started
    current = clock.best.copy()
    weights = {family: 1.0 for family in FAMILIES}
    stagnation = restarts = repairs = 0
    calls = []
    inference_seconds = candidate_seconds = 0.0
    while time.perf_counter() < clock.deadline:
        stage_start = time.perf_counter()
        candidates = candidate_regions(
            current, rng, sizes=config.region_sizes, per_family=config.per_family
        )
        candidate_seconds += time.perf_counter() - stage_start
        if selector is not None:
            stage_start = time.perf_counter()
            scores = selector(current, candidates)
            inference_seconds += time.perf_counter() - stage_start
            if len(scores) != len(candidates) or not all(math.isfinite(float(s)) for s in scores):
                raise ValueError("invalid selector scores")
            highest = max(scores)
            region = rng.choice(
                [r for r, s in zip(candidates, scores, strict=True) if s == highest]
            )
        else:
            available = [
                family for family in FAMILIES if any(r.family == family for r in candidates)
            ]
            family = rng.choices(
                available, [weights[f] if method == "adaptive" else 1 for f in available]
            )[0]
            region = rng.choice([r for r in candidates if r.family == family])
        remaining = clock.deadline - time.perf_counter()
        if remaining <= 0:
            break
        previous_best = clock.best.m
        outcome = repair(
            current,
            region.vertices,
            seconds=min(config.repair_seconds, remaining),
            seed=rng.randrange(2**31 - 1),
        )
        for event in outcome.improvements:
            clock.offer(
                Graph.from_edges(n, event["graph_edges"]),
                "repair",
                completed_at=event["completed_at"],
            )
        current = outcome.graph
        repairs += 1
        calls.append(
            {
                "region": region.vertices,
                "family": region.family,
                "gain": outcome.gain,
                "seconds": outcome.seconds,
                "local_optimal": outcome.local_optimal,
                "rounds": outcome.rounds,
                "cuts": outcome.cuts,
                "status": outcome.status,
            }
        )
        reward = 1.0 + 5.0 * outcome.gain / max(0.01, outcome.seconds)
        weights[region.family] = 0.8 * weights[region.family] + 0.2 * reward
        stagnation = 0 if clock.best.m > previous_best else stagnation + 1
        if stagnation >= config.stagnation_repairs and time.perf_counter() < clock.deadline:
            # Identical diversification for every neighborhood selector.
            current = (
                rng.choice(pool).graph.copy() if rng.random() < 0.5 and pool else clock.best.copy()
            )
            selected = set(rng.sample(range(n), min(3, n)))
            for edge in current.edges():
                if set(edge) & selected:
                    current.remove(*edge)
            greedy_fill(current, rng, deadline=clock.deadline)
            clock.offer(current, "diversification")
            stagnation = 0
            restarts += 1
    result = clock.report()
    verify_witness(n, result["edges"])
    result.update(
        method=method,
        n=n,
        seed=seed,
        initialization_seconds=initialization,
        starting_pool=[
            {
                "kind": s.kind,
                "field_order": s.field_order,
                "sha256": s.graph.digest(),
                "edges": s.graph.edges(),
            }
            for s in pool
        ],
        repairs=repairs,
        restarts=restarts,
        repair_calls=calls,
        inference_seconds=inference_seconds,
        candidate_seconds=candidate_seconds,
        configuration=asdict(config),
    )
    return result

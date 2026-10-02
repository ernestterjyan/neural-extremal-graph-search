"""Timed multistart tabu and shared large-neighborhood search trajectories."""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .constructions import starting_pool
from .controls import AdaptiveRegions
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
    tabu_adaptive_penalty: bool = True
    adaptive_exploration: float = 0.15
    adaptive_smoothing: float = 0.2
    adaptive_reward_scale: float = 5.0

    def __post_init__(self):
        if type(self.tabu_adaptive_penalty) is not bool:
            raise ValueError("tabu adaptive-penalty option must be Boolean")
        AdaptiveRegions(
            exploration=self.adaptive_exploration,
            smoothing=self.adaptive_smoothing,
            reward_scale=self.adaptive_reward_scale,
        )
        if not all(
            math.isfinite(v) and v > 0
            for v in [self.seconds, self.repair_seconds, self.tabu_penalty]
        ):
            raise ValueError("finite positive time limits and penalty required")
        if (
            not self.region_sizes
            or any(k < 1 for k in self.region_sizes)
            or len(set(self.region_sizes)) != len(self.region_sizes)
            or self.per_family < 1
        ):
            raise ValueError("invalid region configuration")
        if (
            self.stagnation_repairs < 1
            or self.tabu_restart_iterations < 1
            or not 1 <= self.tabu_tenure_min <= self.tabu_tenure_max
        ):
            raise ValueError("invalid stagnation or tabu configuration")


class SearchClock:
    def __init__(self, n: int, seconds: float, *, started_at: float | None = None):
        if seconds <= 0:
            raise ValueError("positive search budget required")
        self.started = time.perf_counter() if started_at is None else started_at
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


def tabu_search(
    n: int, seed: int, config: SearchConfig, *, started_at: float | None = None
) -> dict:
    clock = SearchClock(n, config.seconds, started_at=started_at)
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
        current_edges = current.m
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
            new_edges = current_edges + (-1 if present else 1)
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
        # Ban exactly the next `tenure` choices, including the next iteration.
        tabu_until[(u, v)] = (
            iterations + 1 + rng.randint(config.tabu_tenure_min, config.tabu_tenure_max)
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
        if config.tabu_adaptive_penalty and iterations % 25 == 0:
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
    started_at: float | None = None,
) -> dict:
    if method not in {"adaptive", "random", "mlp", "gnn"}:
        raise ValueError("unknown neighborhood method")
    if method in {"mlp", "gnn"} and selector is None:
        raise ValueError("learned method requires a selector")
    if method not in {"mlp", "gnn"} and selector is not None:
        raise ValueError("classical method cannot use an undeclared learned selector")
    clock = SearchClock(n, config.seconds, started_at=started_at)
    rng = random.Random(seed)
    candidate_rng = random.Random(seed + 100000001)
    repair_rng = random.Random(seed + 300000001)
    diversification_rng = random.Random(seed + 400000001)
    pool = initialize(n, seed, clock)
    initialization = time.perf_counter() - clock.started
    current = clock.best.copy()
    adaptive = AdaptiveRegions(
        exploration=config.adaptive_exploration,
        smoothing=config.adaptive_smoothing,
        reward_scale=config.adaptive_reward_scale,
    )
    stagnation = restarts = repairs = 0
    calls = []
    inference_seconds = candidate_seconds = selection_seconds = 0.0
    while time.perf_counter() < clock.deadline:
        stage_start = time.perf_counter()
        candidates = candidate_regions(
            current, candidate_rng, sizes=config.region_sizes, per_family=config.per_family
        )
        candidate_seconds += time.perf_counter() - stage_start
        selection_start = time.perf_counter()
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
        elif method == "adaptive":
            region = adaptive.choose(candidates, rng)
        else:
            available = [
                family for family in FAMILIES if any(r.family == family for r in candidates)
            ]
            family = rng.choice(available)
            region = rng.choice([r for r in candidates if r.family == family])
        selection_seconds += time.perf_counter() - selection_start
        remaining = clock.deadline - time.perf_counter()
        if remaining <= 0:
            break
        previous_best = clock.best.m
        repair_seed = repair_rng.randrange(2**31 - 1)
        outcome = repair(
            current,
            region.vertices,
            seconds=min(config.repair_seconds, remaining),
            seed=repair_seed,
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
                "seed": repair_seed,
                "family": region.family,
                "gain": outcome.gain,
                "seconds": outcome.seconds,
                "local_optimal": outcome.local_optimal,
                "rounds": outcome.rounds,
                "cuts": outcome.cuts,
                "status": outcome.status,
            }
        )
        if method == "adaptive":
            adaptive.observe(region, gain=outcome.gain, seconds=outcome.seconds)
        stagnation = 0 if clock.best.m > previous_best else stagnation + 1
        if stagnation >= config.stagnation_repairs and time.perf_counter() < clock.deadline:
            # Identical diversification for every neighborhood selector.
            current = (
                diversification_rng.choice(pool).graph.copy()
                if diversification_rng.random() < 0.5 and pool
                else clock.best.copy()
            )
            selected = set(diversification_rng.sample(range(n), min(3, n)))
            for edge in current.edges():
                if set(edge) & selected:
                    current.remove(*edge)
            greedy_fill(current, diversification_rng, deadline=clock.deadline)
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
        selection_seconds=selection_seconds,
        adaptive_arms=adaptive.report() if method == "adaptive" else [],
        configuration=asdict(config),
    )
    return result

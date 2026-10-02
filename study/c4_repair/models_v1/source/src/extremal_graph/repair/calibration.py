"""Timed construction and perturb/refill calibration, preserving the original pilot."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict

from .constructions import starting_pool
from .graph import greedy_fill, verify_witness
from .search import SearchClock, SearchConfig, initialize


def calibration_search(n: int, seed: int, config: SearchConfig, *, method: str) -> dict:
    if method not in {"construction", "simple_repair"}:
        raise ValueError("unknown calibration method")
    clock = SearchClock(n, config.seconds)
    rng = random.Random(seed)
    pool = initialize(n, seed, clock)
    initialization = time.perf_counter() - clock.started
    current = clock.best.copy()
    iterations = restarts = 0
    while time.perf_counter() < clock.deadline:
        if method == "construction":
            # Rebuild the same mixture with independent seeds, inside the time limit.
            draws = starting_pool(n, rng.randrange(2**31 - 1), deadline=clock.deadline)
            for start in draws:
                clock.offer(start.graph, "construction", completed_at=start.completed_at)
            iterations += 1
            continue
        # The original pilot removes one or two edges, greedily refills, and uses
        # a 200-step cooling schedule. Repeat that schedule under the common clock.
        proposal = current.copy()
        count = min(proposal.m, 1 + int(rng.random() < 0.35))
        for edge in rng.sample(proposal.edges(), count):
            proposal.remove(*edge)
        greedy_fill(proposal, rng, deadline=clock.deadline)
        clock.offer(proposal, "simple_repair")
        change = proposal.m - current.m
        temperature = 0.8 * (1 - (iterations % 200) / 200) + 0.05
        if change >= 0 or rng.random() < math.exp(change / temperature):
            current = proposal
        iterations += 1
        if iterations % 200 == 0 and time.perf_counter() < clock.deadline:
            current = rng.choice(pool).graph.copy() if pool else clock.best.copy()
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
        iterations=iterations,
        restarts=restarts,
        configuration=asdict(config),
        calibration_scope="timed adaptation; original iteration-count pilot remains frozen",
    )
    return result

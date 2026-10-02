"""CP-SAT incident-edge repair with C4 separation and a feasible incumbent.

An OPTIMAL relaxation solution is a local optimum only if independently C4-free.
Full-region repair alone can certify a global optimum; other regions cannot.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .graph import Graph, cycle_cuts, verify_witness


@dataclass
class RepairResult:
    graph: Graph
    initial_edges: int
    seconds: float
    status: str
    local_optimal: bool
    global_optimal: bool
    rounds: int
    cuts: int
    upper_bound: float | None
    improvements: list[dict] = field(default_factory=list)

    @property
    def gain(self) -> int:
        return self.graph.m - self.initial_edges

    def event_records(self) -> list[dict]:
        return [
            {k: v for k, v in event.items() if k != "completed_at"} for event in self.improvements
        ]


def repair(
    graph: Graph,
    region: tuple[int, ...],
    *,
    seconds: float,
    seed: int,
    deterministic_limit: float | None = None,
    max_rounds: int = 1000,
) -> RepairResult:
    started = time.perf_counter()
    deadline = started + seconds
    if not math.isfinite(seconds) or seconds < 0 or max_rounds < 1:
        raise ValueError("invalid repair budget")
    if deterministic_limit is not None and (
        not math.isfinite(deterministic_limit) or deterministic_limit < 0
    ):
        raise ValueError("invalid deterministic repair budget")
    if not region or len(set(region)) != len(region) or any(not 0 <= u < graph.n for u in region):
        raise ValueError("invalid region")
    verify_witness(graph.n, graph.edges())
    incumbent = graph.copy()
    improvements = []
    if time.perf_counter() >= deadline:
        return RepairResult(
            incumbent,
            graph.m,
            time.perf_counter() - started,
            "TIMEOUT_INCUMBENT",
            False,
            False,
            0,
            0,
            None,
            [],
        )
    selected = set(region)
    outside = [u for u in range(graph.n) if u not in selected]
    fixed = Graph.from_edges(
        graph.n, [(u, v) for u, v in graph.edges() if u not in selected and v not in selected]
    )
    model = cp_model.CpModel()
    variables = {
        (u, v): model.new_bool_var(f"e_{u}_{v}")
        for u, v in itertools.combinations(range(graph.n), 2)
        if u in selected or v in selected
    }
    objective = sum(variables.values()) + fixed.m
    model.maximize(objective)
    model.add(objective >= graph.m)
    # Every cycle with exactly one selected vertex is forbidden in advance.
    # Remaining C4 constraints (two or more selected vertices) are separated lazily.
    for u, v in itertools.combinations(outside, 2):
        if fixed.adjacency[u] & fixed.adjacency[v]:
            for r in region:
                model.add(variables[tuple(sorted((r, u)))] + variables[tuple(sorted((r, v)))] <= 1)
    # Strengthen the relaxation with common-neighbor limits through the fixed
    # complement. This excludes every C4 with at most two selected vertices;
    # cycles with three/four selected vertices still require lazy separation.
    for r in region:
        for v in outside:
            model.add(
                sum(variables[tuple(sorted((r, u)))] for u in outside if fixed.has(u, v)) <= 1
            )
    for r, s in itertools.combinations(region, 2):
        paths = []
        for u in outside:
            a = variables[tuple(sorted((r, u)))]
            b = variables[tuple(sorted((s, u)))]
            path = model.new_bool_var(f"path_{r}_{u}_{s}")
            model.add(path <= a)
            model.add(path <= b)
            model.add(path >= a + b - 1)
            paths.append(path)
        model.add(sum(paths) <= 1)
    for edge, variable in variables.items():
        model.add_hint(variable, int(graph.has(*edge)))
    seen_cuts = set()
    rounds = 0
    upper_bound = None
    status_name = "TIMEOUT_INCUMBENT"
    local_optimal = False
    deterministic_remaining = deterministic_limit

    def decode(reader) -> Graph:
        proposal = fixed.copy()
        for edge, variable in variables.items():
            if reader.value(variable):
                proposal.add(*edge)
        return proposal

    def accept(proposal: Graph) -> None:
        nonlocal incumbent
        count = proposal.m
        witness = proposal.edges() if count > incumbent.m else None
        completed_at = time.perf_counter()
        if completed_at >= deadline:
            return
        if count > incumbent.m:
            improvements.append(
                {
                    "seconds": completed_at - started,
                    "edges": count,
                    "graph_edges": witness,
                    "completed_at": completed_at,
                }
            )
        if count >= incumbent.m:
            incumbent = proposal

    class FeasibleIncumbent(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            if time.perf_counter() >= deadline:
                self.stop_search()
                return
            proposal = decode(self)
            if proposal.c4_count() == 0:
                accept(proposal)
            else:
                # Spend the remaining budget on the corrected model immediately,
                # rather than optimizing a known-invalid relaxation to completion.
                self.stop_search()

    while time.perf_counter() < deadline and rounds < max_rounds:
        if deterministic_remaining is not None and deterministic_remaining <= 0:
            break
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed % (2**31 - 1)
        solver.parameters.max_time_in_seconds = max(1e-6, deadline - time.perf_counter())
        if deterministic_remaining is not None:
            solver.parameters.max_deterministic_time = deterministic_remaining
        status = solver.solve(model, FeasibleIncumbent())
        rounds += 1
        if deterministic_remaining is not None:
            deterministic_remaining -= solver.response_proto.deterministic_time
        status_name = solver.status_name(status)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(f"invalid CP-SAT model: {model.validate()}")
        if status == cp_model.INFEASIBLE:
            raise RuntimeError("repair lost its known feasible incumbent")
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            break
        bound = solver.best_objective_bound
        upper_bound = bound if upper_bound is None else min(bound, upper_bound)
        proposal = decode(solver)
        violations = cycle_cuts(proposal)
        if not violations:
            # A late answer is not credited to the search budget.
            accept(proposal)
            local_optimal = status == cp_model.OPTIMAL and incumbent.m == proposal.m
            break
        new_cuts = violations - seen_cuts
        if not new_cuts:
            raise RuntimeError("C4 separation failed to cut an invalid solution")
        for cycle in sorted(new_cuts):
            variable_edges = [variables[e] for e in cycle if e in variables]
            fixed_edges = sum(fixed.has(*e) for e in cycle if e not in variables)
            model.add(sum(variable_edges) <= 3 - fixed_edges)
        seen_cuts |= new_cuts
        model.clear_hints()
        for edge, variable in variables.items():
            model.add_hint(variable, int(incumbent.has(*edge)))
    verify_witness(incumbent.n, incumbent.edges())
    if any(incumbent.has(u, v) != graph.has(u, v) for u, v in itertools.combinations(outside, 2)):
        raise RuntimeError("repair changed the fixed complement")
    return RepairResult(
        incumbent,
        graph.m,
        time.perf_counter() - started,
        status_name,
        local_optimal,
        local_optimal and len(region) == graph.n,
        rounds,
        len(seen_cuts),
        upper_bound,
        improvements,
    )

"""Exhaust tied one-step-look-ahead paths until a counterexample or state cap."""

from __future__ import annotations

import itertools
import json
import sys
from collections import deque
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.baselines import LookaheadPolicy  # noqa: E402
from extremal_graph.diagnostics import balanced_completion  # noqa: E402
from extremal_graph.features import legal_edges_from_state  # noqa: E402
from extremal_graph.graph import GraphState  # noqa: E402
from extremal_graph.study import independent_verify  # noqa: E402
from extremal_graph.utils import sha256_file, write_json  # noqa: E402

STUDY = ROOT / "study/followup"


def _brute_balance(n: int, edges: tuple[tuple[int, int], ...]) -> bool:
    for side in itertools.combinations(range(n), n // 2):
        members = set(side)
        if all((u in members) != (v in members) for u, v in edges):
            return True
    return False


def _path(mask: int, parents: dict[int, tuple[int, int] | None], pairs: list) -> list:
    actions = []
    while parents[mask] is not None:
        previous, index = parents[mask]
        actions.append(pairs[index])
        mask = previous
    return list(reversed(actions))


def search(n: int, max_states: int) -> dict:
    pairs = list(itertools.combinations(range(n), 2))
    positions = {edge: i for i, edge in enumerate(pairs)}
    parents: dict[int, tuple[int, int] | None] = {0: None}
    queue = deque([0])
    examined = 0
    maximal_frontier = 1
    while queue:
        mask = queue.popleft()
        examined += 1
        state = GraphState(n, 2, tuple(e for i, e in enumerate(pairs) if mask >> i & 1))
        if not balanced_completion(state)[0]:
            raise AssertionError("unbalanced state was inserted before its witness was recorded")
        legal = legal_edges_from_state(state)
        if not legal:
            optimum = n * n // 4
            if state.edge_count != optimum or not all(independent_verify(state).values()):
                raise AssertionError("terminal state contradicts balanced-completion invariant")
            continue
        losses = LookaheadPolicy.losses(state, legal)
        smallest = min(losses)
        for edge, loss in zip(legal, losses, strict=True):
            if loss != smallest:
                continue
            index = positions[edge]
            child = mask | (1 << index)
            if child in parents:
                continue
            child_edges = state.edges + (edge,)
            child_state = GraphState(n, 2, child_edges)
            feasible, reason = balanced_completion(child_state)
            if not feasible:
                if _brute_balance(n, child_state.edges):
                    raise AssertionError(
                        "completion diagnostic contradicts independent partition search"
                    )
                graph = nx.empty_graph(n)
                graph.add_edges_from(child_state.edges)
                if nx.is_bipartite(graph) == (reason == "odd_cycle"):
                    raise AssertionError("failure-cause check disagrees with NetworkX")
                brute_loss = len(legal) - len(legal_edges_from_state(child_state))
                if brute_loss != smallest:
                    raise AssertionError("look-ahead loss differs from independent legal-set count")
                prefix = _path(mask, parents, pairs)
                return {
                    "n": n,
                    "status": "reachable_counterexample",
                    "states_examined": examined,
                    "states_discovered": len(parents),
                    "prefix_actions": prefix,
                    "destructive_tied_action": edge,
                    "prefix_edge_count": len(prefix),
                    "loss": smallest,
                    "failure_cause": reason,
                    "independent_balance_check": False,
                }
            if len(parents) >= max_states:
                return {
                    "n": n,
                    "status": "state_cap",
                    "states_examined": examined,
                    "states_discovered": len(parents),
                    "max_states": max_states,
                }
            parents[child] = (mask, index)
            queue.append(child)
        maximal_frontier = max(maximal_frontier, len(queue))
        if examined % 10000 == 0:
            print(f"look-ahead n={n}: {examined} states examined", flush=True)
    return {
        "n": n,
        "status": "exhaustive_no_failure",
        "states_examined": examined,
        "states_discovered": len(parents),
        "maximal_frontier": maximal_frontier,
    }


def main() -> None:
    protocol = json.loads((STUDY / "lookahead_protocol.json").read_text())
    results = []
    for n in protocol["sizes"]:
        value = search(n, protocol["max_states_per_size"])
        results.append(value)
        print(value, flush=True)
        if value["status"] == "reachable_counterexample":
            break
    output = {
        "protocol_sha256": sha256_file(STUDY / "lookahead_protocol.json"),
        "script_sha256": sha256_file(__file__),
        "results": results,
    }
    write_json(STUDY / "lookahead_reachability.json", output)


if __name__ == "__main__":
    main()

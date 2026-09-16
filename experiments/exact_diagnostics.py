"""Exhaustive continuation certificates for small partial graphs, independent of search code."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.features import collate_graph_states, legal_edges_from_state  # noqa: E402
from extremal_graph.graph import GraphState  # noqa: E402
from extremal_graph.policy import EdgePolicy, action_probabilities  # noqa: E402
from extremal_graph.serialization import graph_to_dict  # noqa: E402
from extremal_graph.study import structural_groups  # noqa: E402


def cycle(n):
    return [(i, (i + 1) % n) for i in range(n)]


def exact_values(state):
    pairs = list(itertools.combinations(range(state.n), 2))
    lookup = {e: 1 << i for i, e in enumerate(pairs)}
    initial = sum(lookup[e] for e in state.edges)
    triangles = [
        lookup[a, b] | lookup[a, c] | lookup[b, c]
        for a, b, c in itertools.combinations(range(state.n), 3)
    ]
    # Legality calculated from independent full triangle masks.
    candidates = [
        e
        for e in pairs
        if e not in state.edges and not any(((initial | lookup[e]) & t) == t for t in triangles)
    ]
    if len(candidates) > 18:
        raise ValueError("exhaustive collection exceeds declared tractability bound")
    values = [-1] * len(candidates)
    witnesses = [None] * len(candidates)
    feasible_subsets = 0
    maximum = state.edge_count
    for subset in range(1 << len(candidates)):
        full = initial
        for j, e in enumerate(candidates):
            if subset >> j & 1:
                full |= lookup[e]
        if any(full & t == t for t in triangles):
            continue
        feasible_subsets += 1
        count = full.bit_count()
        maximum = max(maximum, count)
        for j in range(len(candidates)):
            if subset >> j & 1 and count > values[j]:
                values[j] = count
                witnesses[j] = [list(e) for e in pairs if full & lookup[e]]
    for q, edges in zip(values, witnesses, strict=True):
        graph = nx.Graph()
        graph.add_nodes_from(range(state.n))
        graph.add_edges_from(edges)
        assert not any(nx.triangles(graph).values()) and graph.number_of_edges() == q
        assert set(state.edges).issubset(graph.edges)
    assert candidates == legal_edges_from_state(state)
    groups = structural_groups(state, candidates, 3, "gnn")
    grouped = {}
    for j, key in enumerate(groups):
        grouped.setdefault(str(key), []).append(j)
    return {
        "graph": graph_to_dict(state),
        "enumerated_subsets": 1 << len(candidates),
        "feasible_subsets": feasible_subsets,
        "maximum_edges": maximum,
        "actions": [
            {"edge": e, "exact_continuation_max": q, "witness_edges": w}
            for e, q, w in zip(candidates, values, witnesses, strict=True)
        ],
        "structural_action_groups": list(grouped.values()),
        "mixed_value_groups": sum(len({values[i] for i in ids}) > 1 for ids in grouped.values()),
    }


def main():
    torch.set_num_threads(1)
    collection = {
        "C6": GraphState(6, 2, tuple(cycle(6))),
        "C7": GraphState(7, 2, tuple(cycle(7))),
        "C8": GraphState(8, 2, tuple(cycle(8))),
        "two_C4": GraphState(8, 2, tuple(cycle(4) + [(u + 4, v + 4) for u, v in cycle(4)])),
        "C4_two_isolates": GraphState(6, 2, tuple(cycle(4))),
        "matching3": GraphState(6, 2, ((0, 1), (2, 3), (4, 5))),
    }
    values = {name: exact_values(state) for name, state in collection.items()}
    state = collection["C8"]
    legal = legal_edges_from_state(state)
    batch = collate_graph_states([state], [legal])
    q = [a["exact_continuation_max"] for a in values["C8"]["actions"]]
    assert q.count(16) == 8 and q.count(13) == 4
    responses = []
    for layers in [0, 1, 3, 8]:
        for seed in range(5):
            torch.manual_seed(seed)
            model = EdgePolicy(message_passing_layers=layers).eval()
            with torch.inference_mode():
                logits = model(batch)
                probs, _ = action_probabilities(logits, candidate_mask=batch.candidate_mask)
            responses.append(
                {
                    "layers": layers,
                    "initialization_seed": seed,
                    "logit_spread": float(logits.max() - logits.min()),
                    "bad_action_mass": sum(float(probs[0, i]) for i, v in enumerate(q) if v == 13),
                }
            )
    output = {
        "collection": values,
        "C8_model_checks": responses,
        "C8_conditional_expected_edges_upper_bound": 15,
        "interpretation": "Conditional state limitation only; no global empty-graph ceiling.",
        "certificate": (
            "All 2^k initially legal edge subsets checked against all triangle masks. "
            "Independent NetworkX witness checks."
        ),
    }
    path = ROOT / "study/exact_diagnostics.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print({k: (v["maximum_edges"], v["mixed_value_groups"]) for k, v in values.items()})


if __name__ == "__main__":
    main()

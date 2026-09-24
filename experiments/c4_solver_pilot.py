"""Frozen exploratory baselines for the maximum-edge C4-free graph problem."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study/c4_pilot"
PROTOCOL = STUDY / "protocol.json"
RESULTS = STUDY / "results.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_count(adjacency: list[int]) -> int:
    return sum(bits.bit_count() for bits in adjacency) // 2


def edges(adjacency: list[int]) -> list[tuple[int, int]]:
    return [
        (u, v)
        for u, bits in enumerate(adjacency)
        for v in range(u + 1, len(adjacency))
        if bits & (1 << v)
    ]


def add(adjacency: list[int], edge: tuple[int, int]) -> None:
    u, v = edge
    adjacency[u] |= 1 << v
    adjacency[v] |= 1 << u


def remove(adjacency: list[int], edge: tuple[int, int]) -> None:
    u, v = edge
    adjacency[u] &= ~(1 << v)
    adjacency[v] &= ~(1 << u)


def legal_edges(adjacency: list[int]) -> list[tuple[int, int]]:
    """A new edge closes C4 iff its endpoints already have a length-three path."""
    n = len(adjacency)
    two_hop = [0] * n
    for u, neighbors in enumerate(adjacency):
        while neighbors:
            bit = neighbors & -neighbors
            two_hop[u] |= adjacency[bit.bit_length() - 1]
            neighbors -= bit
    return [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if not adjacency[u] & (1 << v) and not two_hop[u] & adjacency[v]
    ]


def choose(
    adjacency: list[int], candidates: list[tuple[int, int]], method: str, rng: random.Random
) -> tuple[int, int]:
    if method == "uniform":
        return rng.choice(candidates)
    degrees = [bits.bit_count() for bits in adjacency]
    if method == "min_degree":
        scores = [degrees[u] + degrees[v] for u, v in candidates]
        best = min(scores)
        return rng.choice(
            [edge for edge, score in zip(candidates, scores, strict=True) if score == best]
        )
    if method == "lookahead16":
        sampled = rng.sample(candidates, min(16, len(candidates)))
        scores = []
        for edge in sampled:
            add(adjacency, edge)
            scores.append(len(legal_edges(adjacency)))
            remove(adjacency, edge)
        best = max(scores)
        return rng.choice(
            [edge for edge, score in zip(sampled, scores, strict=True) if score == best]
        )
    raise ValueError(f"unsupported edge-choice policy: {method}")


def complete(adjacency: list[int], method: str, rng: random.Random) -> list[int]:
    while candidates := legal_edges(adjacency):
        add(adjacency, choose(adjacency, candidates, method, rng))
    return adjacency


def polarity_graph(q: int) -> list[int]:
    """Orthogonal-polarity graph on PG(2,q), here only for prime q."""
    points = set()
    for vector in itertools.product(range(q), repeat=3):
        if vector == (0, 0, 0):
            continue
        leading = next(value for value in vector if value)
        inverse = pow(leading, -1, q)
        points.add(tuple(value * inverse % q for value in vector))
    vertices = sorted(points)
    if len(vertices) != q * q + q + 1:
        raise AssertionError("projective point count mismatch")
    adjacency = [0] * len(vertices)
    for u, v in itertools.combinations(range(len(vertices)), 2):
        if sum(a * b for a, b in zip(vertices[u], vertices[v], strict=True)) % q == 0:
            add(adjacency, (u, v))
    return adjacency


def polarity_seed(n: int, rng: random.Random) -> tuple[list[int], int]:
    q = next(q for q in [2, 3, 5, 7, 11] if q * q + q + 1 >= n)
    source = polarity_graph(q)
    remaining = set(range(len(source)))
    while len(remaining) > n:
        mask = sum(1 << v for v in remaining)
        degrees = {u: (source[u] & mask).bit_count() for u in remaining}
        smallest = min(degrees.values())
        remaining.remove(rng.choice(sorted(u for u in remaining if degrees[u] == smallest)))
    ordered = sorted(remaining)
    index = {old: new for new, old in enumerate(ordered)}
    adjacency = [0] * n
    for old_u in ordered:
        for old_v in ordered:
            if old_u < old_v and source[old_u] & (1 << old_v):
                add(adjacency, (index[old_u], index[old_v]))
    return adjacency, q


def improve(adjacency: list[int], rng: random.Random) -> list[int]:
    """Time-bounded perturb-and-refill search with a cooling acceptance rule."""
    current = adjacency.copy()
    best = current.copy()
    for attempt in range(200):
        proposal = current.copy()
        for edge in rng.sample(edges(proposal), 1 + int(rng.random() < 0.35)):
            remove(proposal, edge)
        complete(proposal, "min_degree", rng)
        change = edge_count(proposal) - edge_count(current)
        temperature = 0.8 * (1 - attempt / 200) + 0.05
        if change >= 0 or rng.random() < math.exp(change / temperature):
            current = proposal
        if edge_count(proposal) > edge_count(best):
            best = proposal.copy()
    return best


def solve(n: int, method: str, seed: int) -> dict:
    rng = random.Random(seed)
    start = time.perf_counter()
    q = None
    if method in {"polarity", "polarity_swap200"}:
        adjacency, q = polarity_seed(n, rng)
        complete(adjacency, "min_degree", rng)
    else:
        adjacency = complete([0] * n, "min_degree" if method == "swap200" else method, rng)
    if method in {"swap200", "polarity_swap200"}:
        adjacency = improve(adjacency, rng)
    return {
        "n": n,
        "method": method,
        "seed": seed,
        "edge_count": edge_count(adjacency),
        "edges": edges(adjacency),
        "polarity_order": q,
        "seconds": time.perf_counter() - start,
    }


def verify_graph(n: int, edge_list: list[list[int] | tuple[int, int]]) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edge_list)
    if (
        graph.number_of_nodes() != n
        or graph.number_of_edges() != len(edge_list)
        or any(u == v or not (0 <= u < v < n) for u, v in edge_list)
    ):
        raise ValueError("invalid edge, duplicate, or loop in result")
    for u, v in itertools.combinations(range(n), 2):
        if len(set(graph[u]) & set(graph[v])) > 1:
            raise ValueError("result contains a four-cycle")
        if graph.has_edge(u, v):
            continue
        if not any(graph.has_edge(a, b) for a in graph[u] for b in graph[v]):
            raise ValueError("result is not maximal")


def run() -> None:
    if RESULTS.exists():
        raise FileExistsError("pilot results already exist; do not overwrite inspected draws")
    protocol = json.loads(PROTOCOL.read_text())
    records = []
    for n in protocol["sizes"]:
        for method in protocol["methods"]:
            for replication in protocol["replications"]:
                seed = protocol["seed_offset"] + n * 10000 + replication
                result = solve(n, method, seed)
                verify_graph(n, result["edges"])
                result.update(
                    replication=replication,
                    known_reference=protocol["known_exact_values"][str(n)],
                    protocol_sha256=sha(PROTOCOL),
                    script_sha256=sha(Path(__file__)),
                )
                records.append(result)
                print(f"n={n} {method} {replication}: {result['edge_count']}", flush=True)
    temporary = RESULTS.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    temporary.replace(RESULTS)


def verify() -> list[dict]:
    protocol = json.loads(PROTOCOL.read_text())
    records = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    expected = {
        (n, method, replication)
        for n in protocol["sizes"]
        for method in protocol["methods"]
        for replication in protocol["replications"]
    }
    found = {(r["n"], r["method"], r["replication"]) for r in records}
    if len(records) != len(expected) or found != expected:
        raise ValueError("pilot has missing or duplicate evaluation cells")
    for record in records:
        if record["protocol_sha256"] != sha(PROTOCOL) or record["script_sha256"] != sha(
            Path(__file__)
        ):
            raise ValueError("pilot evidence/source contract mismatch")
        if record["edge_count"] != len(record["edges"]):
            raise ValueError("pilot edge count mismatch")
        if record["known_reference"] != protocol["known_exact_values"][str(record["n"])]:
            raise ValueError("pilot reference changed")
        if record["edge_count"] > record["known_reference"]:
            raise ValueError("pilot result exceeds published exact value; investigate")
        verify_graph(record["n"], record["edges"])
    return records


def report() -> None:
    records = verify()
    output = STUDY / "summary.csv"
    rows = []
    for n, method in itertools.product(
        sorted({r["n"] for r in records}), sorted({r["method"] for r in records})
    ):
        subset = [r for r in records if r["n"] == n and r["method"] == method]
        values = [r["edge_count"] for r in subset]
        seconds = [r["seconds"] for r in subset]
        reference = subset[0]["known_reference"]
        rows.append(
            {
                "n": n,
                "method": method,
                "known_exact": reference,
                "mean_edges": sum(values) / len(values),
                "max_edges": max(values),
                "mean_gap": reference - sum(values) / len(values),
                "exact_runs": sum(value == reference for value in values),
                "replications": len(values),
                "mean_seconds": sum(seconds) / len(seconds),
            }
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "verify", "report"])
    args = parser.parse_args()
    if args.stage == "run":
        run()
    elif args.stage == "verify":
        print({"graphs": len(verify()), "valid": True})
    else:
        report()


if __name__ == "__main__":
    main()

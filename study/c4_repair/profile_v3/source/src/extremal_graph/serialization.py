"""Stable JSON serialization for generated graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import GraphState

SCHEMA_VERSION = 1


def graph_to_dict(graph: GraphState) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "n": graph.n,
        "r": graph.r,
        "edges": [[u, v] for u, v in graph.edges],
    }


def graph_from_dict(value: dict[str, Any]) -> GraphState:
    required = {"schema_version", "n", "r", "edges"}
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        raise ValueError(f"missing graph fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown graph fields: {sorted(extra)}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {value['schema_version']!r}")
    if not isinstance(value["edges"], list):
        raise ValueError("edges must be a list")
    edges: list[tuple[int, int]] = []
    for edge in value["edges"]:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("each edge must be a two-element list")
        edges.append((edge[0], edge[1]))
    return GraphState(n=value["n"], r=value["r"], edges=tuple(edges))


def save_graph(graph: GraphState, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph_to_dict(graph), indent=2) + "\n", encoding="utf-8")


def load_graph(path: str | Path) -> GraphState:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid graph JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("serialized graph must be a JSON object")
    return graph_from_dict(value)

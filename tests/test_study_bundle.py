import gzip
import json

import pandas as pd
import pytest

from extremal_graph.study import collect, evaluate_cell, verify_bundle


def test_atomic_evaluation_resume_and_independent_verification(tmp_path):
    params = dict(
        method="untrained_gnn",
        seed=4,
        n=5,
        episodes=7,
        batch_size=3,
        diagnostic_episodes=2,
        diagnostic_sizes=(5,),
    )
    partial = evaluate_cell(output=tmp_path / "resumed", stop_after_chunks=1, **params)
    first = (partial / "chunk-000000.json.gz").read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        verify_bundle(tmp_path / "resumed")
    evaluate_cell(output=tmp_path / "resumed", **params)
    assert (partial / "chunk-000000.json.gz").read_bytes() == first
    full = evaluate_cell(output=tmp_path / "fresh", **params)
    for a in sorted(partial.glob("chunk*.gz")):
        with gzip.open(a, "rt") as handle:
            left = json.load(handle)["records"]
        with gzip.open(full / a.name, "rt") as handle:
            right = json.load(handle)["records"]
        for x, y in zip(left, right, strict=True):
            x["row"].pop("inference_time_seconds")
            y["row"].pop("inference_time_seconds")
            assert x == y
    assert verify_bundle(tmp_path / "resumed")["graphs"] == 7
    path = collect(tmp_path / "resumed", tmp_path / "summary.csv")
    assert len(pd.read_csv(path)) == 7
    with pytest.raises(ValueError, match="contract"):
        evaluate_cell(output=tmp_path / "resumed", **(params | {"episodes": 8}))
    target = partial / "chunk-000000.json.gz"
    target.write_bytes(target.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="hash"):
        verify_bundle(tmp_path / "resumed")

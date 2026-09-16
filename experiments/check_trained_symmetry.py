"""Check the mathematical symmetry predictions on all corrected GNN checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extremal_graph.features import collate_graph_states, legal_edges_from_state  # noqa: E402
from extremal_graph.graph import GraphState  # noqa: E402
from extremal_graph.policy import action_probabilities  # noqa: E402
from extremal_graph.training import load_model_checkpoint  # noqa: E402
from extremal_graph.turan import construct_turan  # noqa: E402
from extremal_graph.utils import sha256_file, write_json  # noqa: E402


def main():
    torch.set_num_threads(1)
    c8 = GraphState(8, 2, tuple((i, (i + 1) % 8) for i in range(8)))
    edges = legal_edges_from_state(c8)
    batch = collate_graph_states([c8], [edges])
    records = []
    for seed in range(5):
        path = ROOT / f"study/artifacts/training/corrected-gnn-seed-{seed}/best.pt"
        model, _ = load_model_checkpoint(path)
        with torch.inference_mode():
            logits = model(batch)
            probabilities, _ = action_probabilities(logits, candidate_mask=batch.candidate_mask)
            terminal = {}
            for n in [6, 14, 24]:
                state = construct_turan(n, 2)
                b = collate_graph_states([state], [[]])
                h = model.node_embeddings(b)[0]
                terminal[str(n)] = float((h - h[0]).abs().max())
        records.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(path),
                "C8_logit_spread": float(logits.max() - logits.min()),
                "C8_bad_action_mass": sum(
                    float(probabilities[0, i]) for i, (u, v) in enumerate(edges) if v - u == 4
                ),
                "balanced_terminal_embedding_max_difference": terminal,
            }
        )
    assert all(r["C8_logit_spread"] < 1e-5 for r in records)
    assert all(abs(r["C8_bad_action_mass"] - 1 / 3) < 1e-6 for r in records)
    assert all(
        all(x == 0 for x in r["balanced_terminal_embedding_max_difference"].values())
        for r in records
    )
    write_json(ROOT / "study/trained_symmetry_checks.json", records)
    print(records)


if __name__ == "__main__":
    main()

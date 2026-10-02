"""A real regular-graph blind spot and permutation/gradient checks for its revision."""

import importlib.util
from pathlib import Path

import pytest
import torch

from extremal_graph.repair.graph import Graph
from extremal_graph.repair.regions import Region
from extremal_graph.repair.selectors import RegionGNN, features

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/c4_repair_region_gnn.py"
spec = importlib.util.spec_from_file_location("marked_gnn", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cycle():
    return Graph.from_edges(8, [(i, (i + 1) % 8) for i in range(8)])


def test_membership_messages_can_distinguish_joint_region_geometry():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    graph = cycle()
    regions = [Region((0, 2), "random"), Region((0, 4), "random")]
    tensors = features(graph, regions)
    # Equal per-node structural features and no internal selected edges mean the
    # current unmarked GNN cannot distinguish these sets, regardless of its weights.
    assert torch.equal(tensors[2][0], tensors[2][1])
    previous = RegionGNN()(*tensors, regions)
    assert previous[0].item() == pytest.approx(previous[1].item(), abs=1e-7)
    model = module.MarkedRegionGNN()
    scores = model(*tensors, regions)
    assert abs((scores[0] - scores[1]).item()) > 1e-5
    # This proves an expressive distinction, not a better repair outcome.


def test_scores_follow_vertex_and_candidate_permutations():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    graph = cycle()
    regions = [Region((0, 2, 5), "random"), Region((1, 3), "neighborhood")]
    permutation = [3, 6, 1, 7, 4, 0, 2, 5]
    renamed = Graph.from_edges(8, [(permutation[u], permutation[v]) for u, v in graph.edges()])
    renamed_regions = [
        Region(tuple(sorted(permutation[u] for u in region.vertices)), region.family)
        for region in regions
    ]
    model = module.MarkedRegionGNN()
    original = model(*features(graph, regions), regions)
    changed = model(*features(renamed, renamed_regions[::-1]), renamed_regions[::-1])
    assert torch.allclose(original, changed.flip(0), atol=1e-6, rtol=1e-5)


def test_single_and_batched_region_scores_agree_and_gradients_are_finite():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    graph = cycle()
    regions = [Region((0, 2), "random"), Region((0, 4), "random")]
    model = module.MarkedRegionGNN()
    batch = model(*features(graph, regions), regions)
    singles = torch.cat([model(*features(graph, [region]), [region]) for region in regions])
    assert torch.allclose(batch, singles, atol=1e-6)
    batch.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

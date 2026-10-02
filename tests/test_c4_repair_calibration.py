"""Actual timed pilot adaptations share starts and credit feasible witnesses only."""

from extremal_graph.repair.calibration import calibration_search
from extremal_graph.repair.graph import verify_witness
from extremal_graph.repair.search import SearchConfig, neighborhood_search, tabu_search


def test_timed_calibrations_share_starts_with_both_classical_searches():
    config = SearchConfig(seconds=0.15, region_sizes=(2, 3), per_family=1, repair_seconds=0.02)
    results = [
        calibration_search(10, 4321, config, method="construction"),
        calibration_search(10, 4321, config, method="simple_repair"),
        tabu_search(10, 4321, config),
        neighborhood_search(10, 4321, config),
    ]
    assert all(r["starting_pool"] == results[0]["starting_pool"] for r in results)
    assert all(r["starting_pool"] and r["elapsed_seconds"] >= 0.15 for r in results)
    assert all(r["iterations"] > 0 for r in results[:3])
    for result in results:
        assert verify_witness(10, result["edges"], explicit_cycles=True)["c4_free"]
        assert result["edges"] == result["history"][-1]["graph_edges"]
        assert all(0 <= event["seconds"] <= 0.15 for event in result["history"])

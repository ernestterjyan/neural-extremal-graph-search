"""Independent small-state checks of the look-ahead limitation."""

from itertools import combinations

from extremal_graph.baselines import LookaheadPolicy
from extremal_graph.features import legal_edges_from_state
from extremal_graph.graph import GraphState


def test_disconnected_star_is_a_statewise_lookahead_counterexample():
    state = GraphState(6, 2, ((0, 1), (0, 2), (0, 3)))
    legal = legal_edges_from_state(state)
    losses = LookaheadPolicy.losses(state, legal)
    chosen = (4, 5)
    child = GraphState(6, 2, state.edges + (chosen,))
    assert losses[legal.index(chosen)] == min(losses) == 1
    assert len(legal) - len(legal_edges_from_state(child)) == 1

    def balanced_cut_exists(graph):
        return any(
            all((u in side) != (v in side) for u, v in graph.edges)
            for chosen_side in combinations(range(6), 3)
            for side in [set(chosen_side)]
        )

    assert balanced_cut_exists(state)
    assert not balanced_cut_exists(child)

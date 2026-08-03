"""Non-learned construction policies."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol

from .env import GraphConstructionEnv
from .graph import Edge, GraphState, Trajectory
from .turan import turan_edge_count


class BaselinePolicy(Protocol):
    def select_action(
        self,
        state: GraphState,
        legal_edges: Sequence[Edge],
        rng: random.Random,
    ) -> Edge: ...


class UniformRandomPolicy:
    def select_action(
        self,
        state: GraphState,
        legal_edges: Sequence[Edge],
        rng: random.Random,
    ) -> Edge:
        del state
        if not legal_edges:
            raise ValueError("cannot select from an empty legal-action set")
        return rng.choice(legal_edges)


class LeastDegreePolicy:
    def select_action(
        self,
        state: GraphState,
        legal_edges: Sequence[Edge],
        rng: random.Random,
    ) -> Edge:
        if not legal_edges:
            raise ValueError("cannot select from an empty legal-action set")
        degrees = state.degrees()
        best_score = min(degrees[u] + degrees[v] for u, v in legal_edges)
        tied = [(u, v) for u, v in legal_edges if degrees[u] + degrees[v] == best_score]
        return rng.choice(tied)


def run_baseline_episode(
    n: int,
    policy: BaselinePolicy,
    *,
    seed: int,
    r: int = 2,
) -> Trajectory:
    env = GraphConstructionEnv(n=n, r=r, seed=seed)
    states: list[GraphState] = []
    actions: list[Edge] = []
    rewards: list[float] = []

    while True:
        legal = env.legal_edges()
        if not legal:
            break
        state = env.state
        action = policy.select_action(state, legal, env.rng)
        transition = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(transition.reward)

    final_state = env.state
    ratio = final_state.edge_count / turan_edge_count(n, r)
    return Trajectory(
        n=n,
        r=r,
        states=tuple(states),
        actions=tuple(actions),
        rewards=tuple(rewards),
        final_state=final_state,
        optimality_ratio=ratio,
    )

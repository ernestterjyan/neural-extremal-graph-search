"""Neural-policy episode sampling."""

from __future__ import annotations

import torch
from torch import nn

from .env import GraphConstructionEnv
from .features import collate_graph_states
from .graph import Edge, GraphState, Trajectory
from .policy import sample_actions
from .turan import turan_edge_count


def run_episode_batch(
    policy: nn.Module,
    *,
    n: int,
    seeds: list[int] | tuple[int, ...],
    r: int = 2,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    sampling_protocol: str = "episode-v2",
) -> list[Trajectory]:
    """Sample complete episodes while batching all currently active graphs."""
    if sampling_protocol not in {"episode-v2", "legacy-batch-v1"}:
        raise ValueError("unknown sampling protocol")
    if not seeds:
        return []
    environments = [GraphConstructionEnv(n=n, r=r, seed=seed) for seed in seeds]
    states_by_episode: list[list[GraphState]] = [[] for _ in seeds]
    actions_by_episode: list[list[Edge]] = [[] for _ in seeds]
    rewards_by_episode: list[list[float]] = [[] for _ in seeds]
    active = list(range(len(environments)))
    generator = torch.Generator(device="cpu")
    combined_seed = sum((index + 1) * seed for index, seed in enumerate(seeds))
    generator.manual_seed(combined_seed % (2**63 - 1))

    episode_generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]

    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            while active:
                active_states = [environments[index].state for index in active]
                legal_by_graph = [environments[index].legal_edges() for index in active]
                still_active = [
                    (episode_index, state, legal)
                    for episode_index, state, legal in zip(
                        active, active_states, legal_by_graph, strict=True
                    )
                    if legal
                ]
                if not still_active:
                    break

                active = [item[0] for item in still_active]
                active_states = [item[1] for item in still_active]
                legal_by_graph = [item[2] for item in still_active]
                batch = collate_graph_states(active_states, legal_by_graph).to(device)
                logits = policy(batch)
                sampled = sample_actions(
                    logits,
                    temperature=temperature,
                    candidate_mask=batch.candidate_mask,
                    generator=generator if sampling_protocol == "legacy-batch-v1" else None,
                    generators=[episode_generators[i] for i in active]
                    if sampling_protocol == "episode-v2"
                    else None,
                )

                next_active: list[int] = []
                for row, episode_index in enumerate(active):
                    candidate_index = int(sampled[row].item())
                    if not 0 <= candidate_index < len(legal_by_graph[row]):
                        raise RuntimeError("invalid sampled index for an active environment")
                    action = legal_by_graph[row][candidate_index]
                    states_by_episode[episode_index].append(active_states[row])
                    actions_by_episode[episode_index].append(action)
                    transition = environments[episode_index].step(action)
                    rewards_by_episode[episode_index].append(transition.reward)
                    if not transition.terminated:
                        next_active.append(episode_index)
                active = next_active

    finally:
        policy.train(was_training)
    trajectories: list[Trajectory] = []
    optimum = turan_edge_count(n, r)
    for index, environment in enumerate(environments):
        final_state = environment.state
        trajectories.append(
            Trajectory(
                n=n,
                r=r,
                states=tuple(states_by_episode[index]),
                actions=tuple(actions_by_episode[index]),
                rewards=tuple(rewards_by_episode[index]),
                final_state=final_state,
                optimality_ratio=final_state.edge_count / optimum,
            )
        )
    return trajectories


def run_episode(
    policy: nn.Module,
    *,
    n: int,
    seed: int,
    r: int = 2,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
    sampling_protocol: str = "episode-v2",
) -> Trajectory:
    return run_episode_batch(
        policy,
        n=n,
        seeds=[seed],
        r=r,
        temperature=temperature,
        device=device,
        sampling_protocol=sampling_protocol,
    )[0]

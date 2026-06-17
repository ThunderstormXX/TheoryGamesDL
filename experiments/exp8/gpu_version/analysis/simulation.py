"""
Seedable, reusable simulation wrapper for vertex-convergence experiments.

This module does **not** re-implement the learning algorithm.  It builds the
existing :class:`BatchedGPUQLearner` / :class:`BatchedGPUSARSALearner` and drives
them with the exact same stateless update loop used in
``supervisor_k_regular_tasks.py`` and ``test_k_regular.py``.  The only additions
are:

* it accepts an **adjacency tensor directly** (so it works with the interpolated
  graphs from :mod:`analysis.interpolation` as well as the ``BaseGraph`` classes);
* it accepts an explicit ``seed`` for reproducibility instead of the hard-coded
  ``42`` used elsewhere;
* it returns a small typed result object with the full per-vertex histories.

Action / reward convention (identical to the rest of the project):
    action == 1  ->  Cooperate   (Q(C) = q[..., 1],  P(C) = softmax(Q/T)[..., 1])
    action == 0  ->  Defect      (Q(D) = q[..., 0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.batched_sarsa import BatchedGPUSARSALearner
from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager

LearnerType = Literal["q_learning", "sarsa"]


@dataclass
class SimulationResult:
    """Container for the output of :func:`run_convergence_simulation`.

    All history arrays have shape ``(T_out, reps, num_nodes)`` where
    ``T_out = iters // record_every + 1`` (the +1 is the initial t=0 snapshot).

    Attributes:
        p_hist:  P(C) per vertex over time.
        qc_hist: Q(Cooperate) per vertex over time.
        qd_hist: Q(Defect) per vertex over time.
        degrees: ``(num_nodes,)`` integer-valued degree of each vertex.
        record_every: number of learning steps between recorded snapshots.
        final_pc: scalar mean P(C) over the last recorded snapshot.
        meta: free-form dictionary of the parameters used.
    """

    p_hist: np.ndarray
    qc_hist: np.ndarray
    qd_hist: np.ndarray
    degrees: np.ndarray
    record_every: int
    final_pc: float
    meta: dict


def run_convergence_simulation(
    adjacency: torch.Tensor,
    *,
    gamma: float = 0.0,
    beta: float = 1.0,
    learner_type: LearnerType = "q_learning",
    iters: int = 200_000,
    reps: int = 256,
    seed: int = 42,
    alpha: float = 0.01,
    record_every: int = 5_000,
    reward_type: str = "pp",
    b: float = 2.0,
    c: float = 1.0,
    bonus: float = 1.0,
    device: torch.device | None = None,
) -> SimulationResult:
    """Run a stateless batched Q-learning / SARSA simulation on one graph.

    Args:
        adjacency: ``(N, N)`` symmetric adjacency matrix (float tensor, 0/1).
        gamma: discount factor.
        beta: inverse temperature for Boltzmann action selection (T = 1/beta).
        learner_type: ``"q_learning"`` (off-policy) or ``"sarsa"`` (on-policy).
        iters: number of learning steps.
        reps: number of parallel independent replicates (the batch dimension).
        seed: RNG seed for full reproducibility (numpy + torch).
        alpha: learning rate.
        record_every: snapshot stride (in learning steps).
        reward_type: reward model passed to :class:`BonusRewardManager`
            (``"pp"``, ``"pf"``, ``"ff"``, ``"fp"``).
        b, c, bonus: reward parameters.
        device: torch device; defaults to the global ``gpu_config.device``.

    Returns:
        A :class:`SimulationResult`.
    """
    if device is not None:
        gpu_utils.gpu_config.device = device
    dev = gpu_utils.gpu_config.device

    temp = 1.0 / beta
    np.random.seed(seed)
    torch.manual_seed(seed)

    adj_t = adjacency.to(device=dev, dtype=torch.float32)
    num_nodes = adj_t.shape[0]
    degrees = adj_t.sum(dim=1)

    states = torch.zeros((reps, num_nodes), dtype=torch.long, device=dev)

    common_kwargs = dict(
        batch_size=reps,
        n_agents=num_nodes,
        action_space_size=2,
        learning_rate=alpha,
        discount_factor=gamma,
        exploration_rate=0.0,
        strategy="boltzmann",
        temperature=temp,
        max_states=1,
    )
    if learner_type == "q_learning":
        learner = BatchedGPUQLearner(**common_kwargs)
    elif learner_type == "sarsa":
        learner = BatchedGPUSARSALearner(**common_kwargs)
    else:
        raise ValueError(f"Unknown learner type: {learner_type!r}")

    reward_manager = BonusRewardManager(reward_type=reward_type, b=b, c=c, bonus=bonus)

    adj_batched = adj_t.unsqueeze(0).expand(reps, -1, -1)
    deg_batched = degrees.unsqueeze(0).expand(reps, -1)

    t_out = iters // record_every + 1
    p_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)
    qc_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)
    qd_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)

    def record_state(t_idx: int) -> None:
        q_now = learner.q_table[:, :, 0, :].cpu()
        probs = torch.softmax(q_now / temp, dim=-1).numpy()
        p_hist[t_idx] = probs[..., 1]
        qd_hist[t_idx] = q_now[..., 0].numpy()
        qc_hist[t_idx] = q_now[..., 1].numpy()

    record_state(0)

    with torch.no_grad():
        if learner_type == "sarsa":
            actions = learner.get_actions(states)
            for t in range(1, iters + 1):
                rewards = reward_manager.calculate_rewards(
                    actions.float(), adj_batched, deg_batched)
                next_actions = learner.get_actions(states)
                learner.update(states, actions, rewards, states, next_actions)
                actions = next_actions
                if t % record_every == 0:
                    record_state(t // record_every)
        else:  # q_learning
            for t in range(1, iters + 1):
                actions = learner.get_actions(states)
                rewards = reward_manager.calculate_rewards(
                    actions.float(), adj_batched, deg_batched)
                learner.update(states, actions, rewards, states)
                if t % record_every == 0:
                    record_state(t // record_every)

    final_pc = float(p_hist[-1].mean())

    return SimulationResult(
        p_hist=p_hist,
        qc_hist=qc_hist,
        qd_hist=qd_hist,
        degrees=degrees.cpu().numpy(),
        record_every=record_every,
        final_pc=final_pc,
        meta=dict(
            gamma=gamma, beta=beta, learner_type=learner_type, iters=iters,
            reps=reps, seed=seed, alpha=alpha, record_every=record_every,
            reward_type=reward_type, b=b, c=c, bonus=bonus,
            num_nodes=int(num_nodes), device=str(dev),
        ),
    )

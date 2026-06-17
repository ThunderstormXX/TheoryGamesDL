"""
Seedable, reusable simulation wrapper for vertex-convergence experiments.

This module does **not** re-implement the learning algorithm.  It builds the
existing :class:`BatchedGPUQLearner` / :class:`BatchedGPUSARSALearner` and drives
them with the same stateless update loop used in ``supervisor_k_regular_tasks.py``,
adding only:

* an explicit ``seed`` for reproducibility;
* an adjacency tensor input (so interpolated graphs work too);
* **A100 / large-GPU optimizations** (see below).

A100 optimizations
------------------
1. **Shared-adjacency reward matmul.**  The graph is identical across the batch,
   so instead of the wasteful ``bmm`` over a ``(reps, N, N)`` expanded tensor we
   compute the cooperation pool with a single ``(reps, N) @ (N, N)`` matmul.  This
   removes the largest memory term and is far better suited to the A100's matmul
   units.  Mathematically identical to ``RewardManager`` (verified by parity test).
2. **Reduced recording.**  We never keep the full ``(T_out, reps, N)`` histories
   in memory; at each snapshot we reduce over the replicate axis on the GPU and
   store only per-vertex mean/std ``(T_out, N)``.  Memory then no longer scales
   with ``reps``, which is what makes very large batches (the A100's strength)
   feasible.  ``store_reps="full"`` restores the old behaviour for small debug runs.
3. **TF32 + inference_mode.**  TF32 matmuls (Ampere+) and ``torch.inference_mode``
   give a free speed-up; Q-value accumulation stays in FP32 so the fine Q
   separations we cluster on are preserved.

Action / reward convention (unchanged):
    action == 1  ->  Cooperate   (Q(C) = q[..., 1],  P(C) = softmax(Q/T)[..., 1])
    action == 0  ->  Defect      (Q(D) = q[..., 0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils
from experiments.exp8.gpu_version.core.batched_gpu import BatchedGPUQLearner
from experiments.exp8.gpu_version.core.batched_sarsa import BatchedGPUSARSALearner

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

LearnerType = Literal["q_learning", "sarsa"]
StoreReps = Literal["reduced", "full"]

_BACKEND_CONFIGURED = False

# Approximate device-memory cost per (replicate, vertex) element of the inner
# loop with the fast shared-adjacency reward: the FP32 Q-table (N*2), the long
# state buffer, and roughly ten transient (reps, N) work tensors (softmax,
# Bernoulli draws, rewards, pool, ...).  Used only to *suggest* a batch size.
_BYTES_PER_REP_NODE = 80


def suggest_reps(
    num_nodes: int,
    *,
    device: torch.device | None = None,
    vram_fraction: float = 0.85,
    max_reps: int = 1_000_000,
    min_reps: int = 256,
    multiple_of: int = 256,
) -> int:
    """Suggest a batch size (``reps``) that targets a fraction of free VRAM.

    Designed for large GPUs (e.g. the 40 GB A100): with the reduced-recording
    simulation, device memory scales as ``reps * num_nodes`` (not ``reps * N**2``),
    so a big batch can fill the card and maximise throughput without host-RAM
    blow-up.  On non-CUDA devices it returns ``min_reps`` (CPU/MPS can't query
    free memory reliably).

    Args:
        num_nodes: number of vertices ``N`` in the graph.
        device: torch device; defaults to the global ``gpu_config.device``.
        vram_fraction: target fraction of *free* VRAM to use.
        max_reps: hard upper cap (avoids absurd batches with tiny graphs).
        min_reps: floor / non-CUDA fallback.
        multiple_of: round down to this multiple (kernel-friendly).

    Returns:
        Suggested ``reps``.
    """
    dev = device or gpu_utils.gpu_config.device
    if dev.type != "cuda":
        return min_reps
    try:
        free_bytes, _total = torch.cuda.mem_get_info(dev)
    except Exception:
        return min_reps
    budget = vram_fraction * float(free_bytes)
    raw = int(budget / max(1, num_nodes) / _BYTES_PER_REP_NODE)
    raw = (raw // multiple_of) * multiple_of
    return int(max(min_reps, min(max_reps, raw)))


def configure_torch_backend(device: torch.device) -> None:
    """Enable Ampere/A100-friendly matmul settings (idempotent, CUDA-only).

    TF32 only affects matmul throughput (the reward pool); Q-value accumulation
    remains FP32, so cluster-relevant Q differences are unaffected.
    """
    global _BACKEND_CONFIGURED
    if _BACKEND_CONFIGURED or device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _BACKEND_CONFIGURED = True


@dataclass
class SimulationResult:
    """Output of :func:`run_convergence_simulation`.

    The ``*_mean`` / ``*_std`` arrays have shape ``(T_out, N)`` and are the
    per-vertex replicate mean / std at each recorded snapshot, where
    ``T_out = iters // record_every + 1``.  Full per-replicate histories are
    populated only when ``store_reps="full"``.

    Attributes:
        p_mean/qc_mean/qd_mean: ``(T_out, N)`` replicate-mean trajectories.
        p_std/qc_std/qd_std:    ``(T_out, N)`` replicate-std trajectories.
        degrees: ``(N,)`` vertex degrees.
        record_every: learning steps between snapshots.
        final_pc: scalar mean P(C) at the last snapshot.
        meta: parameters used.
        p_hist/qc_hist/qd_hist: optional full ``(T_out, reps, N)`` arrays.
    """

    p_mean: np.ndarray
    qc_mean: np.ndarray
    qd_mean: np.ndarray
    p_std: np.ndarray
    qc_std: np.ndarray
    qd_std: np.ndarray
    degrees: np.ndarray
    record_every: int
    final_pc: float
    meta: dict
    p_hist: Optional[np.ndarray] = None
    qc_hist: Optional[np.ndarray] = None
    qd_hist: Optional[np.ndarray] = None

    def slice_nodes(self, sl) -> "SimulationResult":
        """Return a view restricted to a contiguous node block ``sl``.

        Used by the graph-fusion path: one simulation runs on a block-diagonal
        super-graph, then each original graph's results are recovered by slicing
        its node range out of every per-vertex array.
        """
        def s2(a):
            return None if a is None else a[:, sl]

        def s3(a):
            return None if a is None else a[:, :, sl]

        degrees = self.degrees[sl]
        meta = {**self.meta, "num_nodes": int(np.asarray(degrees).shape[0]),
                "fused_total_nodes": int(self.meta.get("num_nodes", 0)), "fused": True}
        return SimulationResult(
            p_mean=s2(self.p_mean), qc_mean=s2(self.qc_mean), qd_mean=s2(self.qd_mean),
            p_std=s2(self.p_std), qc_std=s2(self.qc_std), qd_std=s2(self.qd_std),
            degrees=degrees, record_every=self.record_every,
            final_pc=float(self.p_mean[-1, sl].mean()), meta=meta,
            p_hist=s3(self.p_hist), qc_hist=s3(self.qc_hist), qd_hist=s3(self.qd_hist),
        )


def compute_pool(cooperators: torch.Tensor, adjacency: torch.Tensor,
                 degrees: torch.Tensor, reward_type: str) -> torch.Tensor:
    """Cooperation pool ``sum_j A_ij f(x_j)`` via one shared-adjacency matmul.

    ``cooperators`` is ``(reps, N)`` and ``adjacency`` is the single ``(N, N)``
    graph.  Because ``A`` is symmetric, ``x @ A`` equals the per-vertex pool.
    For the fitness-scaled models (``ff``/``fp``) the contribution is divided by
    the neighbour degree first.
    """
    if reward_type in ("ff", "fp"):
        scaled = cooperators / torch.clamp(degrees, min=1.0)
        return scaled @ adjacency
    return cooperators @ adjacency


def compute_rewards_shared(cooperators: torch.Tensor, adjacency: torch.Tensor,
                           degrees: torch.Tensor, *, reward_type: str,
                           b: float, c: float, bonus: float) -> torch.Tensor:
    """Reward for every agent — fast shared-adjacency equivalent of
    :class:`BonusRewardManager`.

    Matches ``RewardManager.calculate_rewards`` for all four models plus the
    constant ``bonus`` term, but with a single ``(reps, N) @ (N, N)`` matmul
    instead of a batched ``(reps, N, N)`` ``bmm``.
    """
    pool = compute_pool(cooperators, adjacency, degrees, reward_type)
    if reward_type in ("pp", "fp"):
        cost = c * cooperators * degrees
    else:  # 'pf', 'ff'
        cost = c * cooperators
    return b * pool - cost + bonus


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
    store_reps: StoreReps = "reduced",
    fast_reward: bool = True,
    progress: bool = False,
    progress_desc: str = "steps",
    progress_position: int = 1,
) -> SimulationResult:
    """Run a stateless batched Q-learning / SARSA simulation on one graph.

    Args:
        adjacency: ``(N, N)`` symmetric adjacency matrix (float tensor, 0/1).
        gamma: discount factor.
        beta: inverse temperature for Boltzmann action selection (T = 1/beta).
        learner_type: ``"q_learning"`` (off-policy) or ``"sarsa"`` (on-policy).
        iters: number of learning steps.
        reps: number of parallel replicates (batch dimension).  On an A100 this
            can be very large (e.g. 4096–32768) thanks to reduced recording.
        seed: RNG seed for reproducibility (numpy + torch).
        alpha: learning rate.
        record_every: snapshot stride (in learning steps).
        reward_type: reward model (``"pp" | "pf" | "ff" | "fp"``).
        b, c, bonus: reward parameters.
        device: torch device; defaults to the global ``gpu_config.device``.
        store_reps: ``"reduced"`` keeps only ``(T_out, N)`` mean/std (memory
            independent of ``reps``); ``"full"`` also keeps the per-replicate
            ``(T_out, reps, N)`` histories.
        fast_reward: use the shared-adjacency matmul reward (recommended).  When
            ``False`` falls back to :class:`BonusRewardManager` (for parity).
        progress: show an inner tqdm bar over the learning steps (steps/s — handy
            for gauging throughput).  Disable it under multiprocessing to avoid
            interleaved bars.
        progress_desc: label for the inner bar.
        progress_position: tqdm line position (use 1 to sit under an outer bar).

    Returns:
        A :class:`SimulationResult`.
    """
    if device is not None:
        gpu_utils.gpu_config.device = device
    dev = gpu_utils.gpu_config.device
    configure_torch_backend(dev)

    temp = 1.0 / beta
    np.random.seed(seed)
    torch.manual_seed(seed)

    adj_t = adjacency.to(device=dev, dtype=torch.float32)
    num_nodes = adj_t.shape[0]
    degrees = adj_t.sum(dim=1)  # (N,)

    states = torch.zeros((reps, num_nodes), dtype=torch.long, device=dev)

    common_kwargs = dict(
        batch_size=reps, n_agents=num_nodes, action_space_size=2,
        learning_rate=alpha, discount_factor=gamma, exploration_rate=0.0,
        strategy="boltzmann", temperature=temp, max_states=1,
    )
    if learner_type == "q_learning":
        learner = BatchedGPUQLearner(**common_kwargs)
    elif learner_type == "sarsa":
        learner = BatchedGPUSARSALearner(**common_kwargs)
    else:
        raise ValueError(f"Unknown learner type: {learner_type!r}")

    # Fallback reward path (kept for parity / fast_reward=False).
    reward_manager = None
    adj_batched = deg_batched = None
    if not fast_reward:
        from experiments.exp8.gpu_version.core.bonus_reward_manager import BonusRewardManager
        reward_manager = BonusRewardManager(reward_type=reward_type, b=b, c=c, bonus=bonus)
        adj_batched = adj_t.unsqueeze(0).expand(reps, -1, -1)
        deg_batched = degrees.unsqueeze(0).expand(reps, -1)

    t_out = iters // record_every + 1
    # Reduced (per-vertex mean/std) snapshots; tiny and reps-independent.
    p_mean = np.zeros((t_out, num_nodes), dtype=np.float32)
    qc_mean = np.zeros((t_out, num_nodes), dtype=np.float32)
    qd_mean = np.zeros((t_out, num_nodes), dtype=np.float32)
    p_std = np.zeros((t_out, num_nodes), dtype=np.float32)
    qc_std = np.zeros((t_out, num_nodes), dtype=np.float32)
    qd_std = np.zeros((t_out, num_nodes), dtype=np.float32)

    keep_full = store_reps == "full"
    p_hist = qc_hist = qd_hist = None
    if keep_full:
        p_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)
        qc_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)
        qd_hist = np.zeros((t_out, reps, num_nodes), dtype=np.float32)

    def record_state(t_idx: int) -> None:
        q_now = learner.q_table[:, :, 0, :]            # (reps, N, 2), on device
        probs_c = torch.softmax(q_now / temp, dim=-1)[..., 1]  # (reps, N)
        qc = q_now[..., 1]
        qd = q_now[..., 0]
        # Reduce over the replicate axis on the GPU, then move tiny (N,) arrays.
        p_mean[t_idx] = probs_c.mean(dim=0).cpu().numpy()
        p_std[t_idx] = probs_c.std(dim=0).cpu().numpy()
        qc_mean[t_idx] = qc.mean(dim=0).cpu().numpy()
        qc_std[t_idx] = qc.std(dim=0).cpu().numpy()
        qd_mean[t_idx] = qd.mean(dim=0).cpu().numpy()
        qd_std[t_idx] = qd.std(dim=0).cpu().numpy()
        if keep_full:
            p_hist[t_idx] = probs_c.cpu().numpy()
            qc_hist[t_idx] = qc.cpu().numpy()
            qd_hist[t_idx] = qd.cpu().numpy()

    def rewards_for(actions: torch.Tensor) -> torch.Tensor:
        cooperators = actions.float()
        if fast_reward:
            return compute_rewards_shared(
                cooperators, adj_t, degrees,
                reward_type=reward_type, b=b, c=c, bonus=bonus)
        return reward_manager.calculate_rewards(cooperators, adj_batched, deg_batched)

    record_state(0)

    step_iter = range(1, iters + 1)
    if progress and tqdm is not None:
        step_iter = tqdm(
            step_iter, total=iters, desc=progress_desc, unit="step",
            unit_scale=True, leave=False, mininterval=0.5,
            position=progress_position, dynamic_ncols=True)

    with torch.inference_mode():
        if learner_type == "sarsa":
            actions = learner.get_actions(states)
            for t in step_iter:
                rewards = rewards_for(actions)
                next_actions = learner.get_actions(states)
                learner.update(states, actions, rewards, states, next_actions)
                actions = next_actions
                if t % record_every == 0:
                    record_state(t // record_every)
        else:  # q_learning
            for t in step_iter:
                actions = learner.get_actions(states)
                rewards = rewards_for(actions)
                learner.update(states, actions, rewards, states)
                if t % record_every == 0:
                    record_state(t // record_every)

    final_pc = float(p_mean[-1].mean())

    return SimulationResult(
        p_mean=p_mean, qc_mean=qc_mean, qd_mean=qd_mean,
        p_std=p_std, qc_std=qc_std, qd_std=qd_std,
        degrees=degrees.cpu().numpy(), record_every=record_every,
        final_pc=final_pc,
        meta=dict(
            gamma=gamma, beta=beta, learner_type=learner_type, iters=iters,
            reps=reps, seed=seed, alpha=alpha, record_every=record_every,
            reward_type=reward_type, b=b, c=c, bonus=bonus,
            num_nodes=int(num_nodes), device=str(dev),
            store_reps=store_reps, fast_reward=fast_reward,
        ),
        p_hist=p_hist, qc_hist=qc_hist, qd_hist=qd_hist,
    )

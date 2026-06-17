"""
Graph fusion — run many graphs in one simulation to saturate a big GPU.

At small batch sizes the per-step kernels are tiny and the simulation is bound by
**kernel-launch latency** over the long ``iters`` loop, leaving an A100 mostly
idle.  Because the learning is stateless and per-agent, several independent
graphs can be packed into a single **block-diagonal** adjacency matrix and run in
one simulation: there are no edges between blocks, so every block evolves exactly
as it would on its own.  This turns ``G`` separate ``iters``-step Python loops
into a single one over a larger node dimension — far fewer launches, much higher
GPU utilisation, identical dynamics (up to RNG interleaving).

Use :func:`block_diagonal_adjacency` to build the super-graph and
``SimulationResult.slice_nodes`` (plus :func:`analysis.pipeline.analyze_from_sim`)
to recover and analyse each original graph afterwards.
"""

from __future__ import annotations

import torch


def block_diagonal_adjacency(
    adjacencies: list[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list[slice]]:
    """Stack adjacency matrices on the block diagonal.

    Args:
        adjacencies: list of ``(N_i, N_i)`` symmetric adjacency tensors.
        device: device for the combined tensor (defaults to the first block's).

    Returns:
        ``(big_adj, ranges)`` where ``big_adj`` is the ``(sum N_i, sum N_i)``
        block-diagonal matrix and ``ranges[i]`` is the ``slice`` of node indices
        belonging to the i-th input graph.
    """
    if not adjacencies:
        raise ValueError("need at least one adjacency matrix to fuse")
    dev = device or adjacencies[0].device
    sizes = [int(a.shape[0]) for a in adjacencies]
    total = sum(sizes)
    big = torch.zeros((total, total), dtype=torch.float32, device=dev)

    ranges: list[slice] = []
    offset = 0
    for adj, size in zip(adjacencies, sizes):
        big[offset:offset + size, offset:offset + size] = adj.to(device=dev, dtype=torch.float32)
        ranges.append(slice(offset, offset + size))
        offset += size
    return big, ranges

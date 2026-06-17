"""
Task 3 — continuous interpolation between a k-regular and a (k+1)-regular graph.

Instead of jumping discretely from ``k-regular`` to ``(k+1)-regular`` by adding
a single edge, we introduce a *topological temperature* ``t in [0, 1]``:

    t = 0  ->  strictly k-regular graph
    t = 1  ->  strictly (k+1)-regular graph

The construction is *mass interpolation of structure*:

1. Build a canonical **k-regular base** graph ``G_k`` (a circulant, matching the
   project's existing ``RingGraph`` / ``*CirculantGraph`` families).
2. Find a set ``E_add`` of edges whose addition turns ``G_k`` into a valid
   (k+1)-regular graph.  Because every vertex must gain exactly one degree,
   ``E_add`` is a **perfect matching** of non-edges (so ``n`` must be even).
3. Add a temperature-controlled fraction of ``E_add``:

   * **Variant A (deterministic):** add the first ``round(t * |E_add|)`` edges.
   * **Variant B (stochastic, default):** add each edge independently with
     probability ``t``; average experiment results over several realizations.

Both endpoints are exact: ``t=0`` adds nothing (k-regular); ``t=1`` adds the
whole matching (k+1-regular).  Intermediate ``t`` yields a mixed
k/(k+1)-regular graph — the analogue of a finite-temperature topology.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch

InterpMode = Literal["stochastic", "deterministic"]


# ════════════════════════════════════════════════════════════════════════════
# Canonical regular base graph (circulant) — reproduces the project's families
# ════════════════════════════════════════════════════════════════════════════

def _circulant_offsets(n: int, k: int) -> list[int]:
    """Connection offsets of a canonical k-regular circulant graph C(n, S).

    * even k -> S = {1, ..., k/2}                      (each offset adds 2 to deg)
    * odd  k -> S = {1, ..., (k-1)/2} U {n/2}          (antipodal adds 1; needs n even)

    This matches RingGraph (k=2), CubicCirculantGraph (k=3),
    QuarticCirculantGraph (k=4) and QuinticCirculantGraph (k=5).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k % 2 == 0:
        return list(range(1, k // 2 + 1))
    # odd k needs the antipodal offset, which requires an even n
    if n % 2 != 0:
        raise ValueError(f"odd k={k} requires even n, got n={n}")
    offsets = list(range(1, (k - 1) // 2 + 1))
    offsets.append(n // 2)
    return offsets


def build_regular_adjacency(
    n: int, k: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Build the canonical k-regular circulant adjacency matrix ``(n, n)``."""
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")
    dev = device or torch.device("cpu")
    adj = torch.zeros((n, n), dtype=torch.float32, device=dev)
    for off in _circulant_offsets(n, k):
        for i in range(n):
            j = (i + off) % n
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


# ════════════════════════════════════════════════════════════════════════════
# Transition edge set E_add (a perfect matching of non-edges)
# ════════════════════════════════════════════════════════════════════════════

def build_transition_matching(adj: torch.Tensor) -> list[tuple[int, int]]:
    """Find a perfect matching consisting only of *non-edges* of ``adj``.

    Adding such a matching raises every vertex's degree by exactly one, turning a
    k-regular graph into a (k+1)-regular one.

    Strategy:
        1. Prefer the **antipodal** matching ``i -- (i + n/2)`` when those are all
           non-edges (this is the natural k->k+1 step for even k).
        2. Otherwise fall back to a deterministic **greedy** matching over the
           complement graph.

    Raises:
        ValueError: if ``n`` is odd, or no perfect matching of non-edges exists.
    """
    a = adj.detach().cpu().numpy()
    n = a.shape[0]
    if n % 2 != 0:
        raise ValueError(f"interpolation requires even n for a perfect matching, got n={n}")

    half = n // 2
    antipodal = [(i, i + half) for i in range(half)]
    if all(a[i, j] == 0 for i, j in antipodal):
        return antipodal

    # Greedy matching over non-edges (deterministic given the adjacency).
    matched = np.zeros(n, dtype=bool)
    edges: list[tuple[int, int]] = []
    for u in range(n):
        if matched[u]:
            continue
        partner = -1
        for v in range(u + 1, n):
            if not matched[v] and a[u, v] == 0:
                partner = v
                break
        if partner == -1:
            raise ValueError(
                f"could not build a perfect matching of non-edges for n={n}; "
                "graph is too dense for a (k+1)-regular target")
        matched[u] = matched[partner] = True
        edges.append((u, partner))
    return edges


# ════════════════════════════════════════════════════════════════════════════
# The interpolating generator
# ════════════════════════════════════════════════════════════════════════════

def generate_interpolated_regular_graph(
    n: int,
    k: int,
    temperature: float,
    seed: Optional[int] = None,
    *,
    mode: InterpMode = "stochastic",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a graph on the continuum between k-regular and (k+1)-regular.

    Args:
        n: number of vertices (must be even).
        k: base regularity; the target is ``k+1``.
        temperature: ``t in [0, 1]``.  ``0`` -> k-regular, ``1`` -> (k+1)-regular.
        seed: RNG seed (used only in ``"stochastic"`` mode) for reproducibility.
        mode: ``"deterministic"`` (Variant A) adds the first ``round(t*|E_add|)``
            matching edges; ``"stochastic"`` (Variant B) adds each matching edge
            independently with probability ``t``.
        device: torch device for the returned tensor.

    Returns:
        ``(n, n)`` float adjacency matrix.
    """
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"temperature must be in [0, 1], got {temperature}")

    dev = device or torch.device("cpu")
    adj = build_regular_adjacency(n, k, device=dev)
    e_add = build_transition_matching(adj)
    m = len(e_add)

    if mode == "deterministic":
        n_add = int(round(temperature * m))
        chosen = e_add[:n_add]
    elif mode == "stochastic":
        rng = np.random.default_rng(seed)
        draws = rng.random(m)
        chosen = [e for e, p in zip(e_add, draws) if p < temperature]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    for i, j in chosen:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def expected_added_edges(n: int, k: int, temperature: float) -> float:
    """Expected number of added edges at a given temperature (size of E_add * t)."""
    adj = build_regular_adjacency(n, k)
    return len(build_transition_matching(adj)) * float(temperature)

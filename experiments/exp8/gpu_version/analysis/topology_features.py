"""
Per-vertex structural (topology) descriptors.

These descriptors let us correlate the *convergence class* of a vertex (found by
clustering its final Q-values, see :mod:`analysis.convergence_clustering`) with
its *structural position* in the graph, as requested in the task:

    degree
    clustering_coefficient
    betweenness_centrality
    eigenvector_centrality

All features are returned as ``numpy`` arrays indexed by node id ``0..N-1`` so
they align positionally with the simulation's per-vertex arrays.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is expected to be installed
    nx = None

# Canonical ordering of structural features (used for CSV columns / correlations).
TOPOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "degree",
    "clustering_coefficient",
    "betweenness_centrality",
    "eigenvector_centrality",
)


def _to_numpy_adjacency(adjacency) -> np.ndarray:
    """Coerce a torch tensor / numpy array adjacency matrix to a float ndarray."""
    if isinstance(adjacency, torch.Tensor):
        adj = adjacency.detach().cpu().numpy()
    else:
        adj = np.asarray(adjacency)
    return adj.astype(np.float64)


def adjacency_to_graph(adjacency) -> "nx.Graph":
    """Build an undirected :class:`networkx.Graph` from an adjacency matrix.

    All ``N`` nodes ``0..N-1`` are added explicitly so that isolated vertices are
    preserved and node indexing matches the simulation arrays.
    """
    if nx is None:
        raise ImportError("networkx is required for topology features")
    adj = _to_numpy_adjacency(adjacency)
    n = adj.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    g.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return g


def _eigenvector_centrality(g: "nx.Graph", n: int) -> np.ndarray:
    """Eigenvector centrality with robust fallbacks for degenerate graphs."""
    try:
        ec = nx.eigenvector_centrality_numpy(g)
        return np.array([ec.get(i, 0.0) for i in range(n)], dtype=np.float64)
    except Exception:
        # Power-iteration fallback, then a uniform fallback if even that fails
        # (e.g. empty graph). This keeps the pipeline from crashing on edge cases.
        try:
            ec = nx.eigenvector_centrality(g, max_iter=1000, tol=1e-6)
            return np.array([ec.get(i, 0.0) for i in range(n)], dtype=np.float64)
        except Exception:
            return np.zeros(n, dtype=np.float64)


def compute_topology_features(adjacency) -> dict[str, np.ndarray]:
    """Compute the four structural descriptors for every vertex.

    Args:
        adjacency: ``(N, N)`` adjacency matrix (torch tensor or numpy array).

    Returns:
        Dict mapping each name in :data:`TOPOLOGY_FEATURE_NAMES` to an
        ``(N,)`` ``float64`` array.  ``degree`` is the raw integer degree
        (stored as float); centralities use networkx's normalized definitions.
    """
    g = adjacency_to_graph(adjacency)
    n = g.number_of_nodes()

    degree = np.array([d for _, d in g.degree()], dtype=np.float64)

    clustering = nx.clustering(g)
    clustering_coefficient = np.array(
        [clustering.get(i, 0.0) for i in range(n)], dtype=np.float64)

    if g.number_of_edges() == 0 or n < 3:
        betweenness = np.zeros(n, dtype=np.float64)
    else:
        bc = nx.betweenness_centrality(g, normalized=True)
        betweenness = np.array([bc.get(i, 0.0) for i in range(n)], dtype=np.float64)

    eigenvector = _eigenvector_centrality(g, n)

    return {
        "degree": degree,
        "clustering_coefficient": clustering_coefficient,
        "betweenness_centrality": betweenness,
        "eigenvector_centrality": eigenvector,
    }


def degree_distribution(adjacency) -> dict[int, int]:
    """Return a ``{degree: count}`` histogram for the graph."""
    adj = _to_numpy_adjacency(adjacency)
    degs = adj.sum(axis=1).round().astype(int)
    values, counts = np.unique(degs, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}

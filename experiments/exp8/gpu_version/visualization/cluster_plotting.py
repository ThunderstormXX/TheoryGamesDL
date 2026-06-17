"""
Visualization for convergence-cluster analysis.

Two figures are provided:

* :func:`plot_convergence_clusters` — draws the graph with NetworkX, colouring
  each vertex by its discovered convergence cluster and labelling it with
  ``id`` / ``degree`` / ``cluster``.
* :func:`plot_q_curves_by_cluster` — the Q(C)/Q(D) trajectories over time,
  coloured by cluster, so the asymptotic separation that motivated the
  clustering is visible at a glance.

These live alongside ``visualization/plotting.py`` and do not modify it.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


# Distinct, colour-blind-friendly palette for cluster ids; noise (-1) is grey.
_CLUSTER_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f",
    "#1abc9c", "#e67e22", "#34495e", "#fd79a8", "#00cec9",
]
_NOISE_COLOR = "#7f8c8d"


def cluster_color_map(cluster_ids: np.ndarray) -> dict[int, str]:
    """Map each cluster id to a stable hex colour (``-1`` -> grey)."""
    uniq = sorted(int(c) for c in np.unique(cluster_ids))
    colors: dict[int, str] = {}
    palette_idx = 0
    for cid in uniq:
        if cid < 0:
            colors[cid] = _NOISE_COLOR
            continue
        if palette_idx < len(_CLUSTER_PALETTE):
            colors[cid] = _CLUSTER_PALETTE[palette_idx]
        else:  # overflow: sample a continuous colormap
            colors[cid] = to_hex(cm.tab20((palette_idx % 20) / 20.0))
        palette_idx += 1
    return colors


def plot_convergence_clusters(
    adjacency,
    cluster_ids: np.ndarray,
    degrees: np.ndarray,
    save_path: str,
    *,
    title: str = "Convergence clusters",
    vertex_ids: Optional[np.ndarray] = None,
    layout: str = "circular",
    seed: int = 42,
    figsize: tuple[float, float] = (9.0, 9.0),
    dpi: int = 150,
) -> str:
    """Draw the graph with vertices coloured by convergence cluster.

    Each vertex is labelled ``id\\nd=<degree>\\nc=<cluster>``.

    Args:
        adjacency: ``(N, N)`` adjacency matrix (torch tensor or numpy array).
        cluster_ids: ``(N,)`` cluster id per vertex (``-1`` = noise).
        degrees: ``(N,)`` vertex degrees.
        save_path: output PNG path.
        title: figure title.
        vertex_ids: optional explicit vertex ids (defaults to ``0..N-1``).
        layout: ``"circular"``, ``"spring"`` or ``"kamada_kawai"``.
        seed: layout seed (spring layout) for reproducibility.

    Returns:
        Absolute path of the written PNG.
    """
    if nx is None:
        raise ImportError("networkx is required for plot_convergence_clusters")

    adj = adjacency.detach().cpu().numpy() if hasattr(adjacency, "detach") else np.asarray(adjacency)
    n = adj.shape[0]
    cluster_ids = np.asarray(cluster_ids).astype(int)
    degrees = np.asarray(degrees).round().astype(int)
    if vertex_ids is None:
        vertex_ids = np.arange(n)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    g.add_edges_from(zip(rows.tolist(), cols.tolist()))

    if layout == "spring":
        pos = nx.spring_layout(g, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g) if g.number_of_edges() else nx.circular_layout(g)
    else:
        pos = nx.circular_layout(g)

    cmap = cluster_color_map(cluster_ids)
    node_colors = [cmap[int(c)] for c in cluster_ids]
    labels = {
        i: f"{int(vertex_ids[i])}\nd={int(degrees[i])}\nc={int(cluster_ids[i])}"
        for i in range(n)
    }

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(g, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=850,
                           edgecolors="black", linewidths=0.6)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7,
                            font_color="white", font_weight="bold")

    # Legend: one entry per cluster id.
    handles = []
    for cid in sorted(cmap):
        lbl = "noise" if cid < 0 else f"cluster {cid}"
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=cmap[cid], markersize=11, label=lbl))
    plt.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=dpi)
    plt.close()
    return abs_path


def _smooth(y: np.ndarray, box_pts: int = 5) -> np.ndarray:
    """Light moving-average smoothing (matches the project's `smooth` helper)."""
    y = np.asarray(y, dtype=float)
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    ys = np.convolve(y, box, mode="same")
    ys[:box_pts] = y[:box_pts]
    ys[-box_pts:] = y[-box_pts:]
    return ys


def plot_q_curves_by_cluster(
    qc_hist: np.ndarray,
    qd_hist: np.ndarray,
    cluster_ids: np.ndarray,
    save_path: str,
    *,
    p_hist: Optional[np.ndarray] = None,
    record_every: int = 5_000,
    title: str = "Convergence by cluster",
    degrees: Optional[np.ndarray] = None,
    dpi: int = 130,
) -> str:
    """Plot per-vertex trajectories coloured by convergence cluster.

    Left panel: the **cooperation probability p(C)** per vertex when ``p_hist``
    is given (this is the most interpretable view); it falls back to Q(C) only
    if ``p_hist`` is omitted.  Right panel: Q(C) (solid) vs Q(D) (dashed).
    Each line is one vertex's replicate-mean trajectory; vertices sharing a
    cluster share a colour, so the limiting levels and their separation are
    directly visible.

    Accepts either the full ``(T_out, reps, N)`` histories from a simulation, or
    the already replicate-averaged ``(T_out, N)`` histories restored from saved
    artifacts — so the figure can be regenerated without re-running anything.

    Args:
        qc_hist, qd_hist: ``(T_out, reps, N)`` or ``(T_out, N)`` Q-value histories.
        cluster_ids: ``(N,)`` cluster id per vertex.
        save_path: output PNG path.
        p_hist: optional ``(T_out, reps, N)`` or ``(T_out, N)`` P(C) history; when
            given, the left panel shows the cooperation probability.
        record_every: snapshot stride (for the x-axis in learning steps).
        title: figure title.
        degrees: optional ``(N,)`` degrees (currently used only for context).

    Returns:
        Absolute path of the written PNG.
    """
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass

    def _to_mean(arr):
        a = np.asarray(arr)
        return a.mean(axis=1) if a.ndim == 3 else a  # (T_out, N)

    qc_mean = _to_mean(qc_hist)
    qd_mean = _to_mean(qd_hist)
    p_mean = _to_mean(p_hist) if p_hist is not None else None
    show_p = p_mean is not None

    t_out, n = qc_mean.shape
    x = np.arange(t_out) * int(record_every)
    cluster_ids = np.asarray(cluster_ids).astype(int)
    cmap = cluster_color_map(cluster_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    legended: set[int] = set()

    left = p_mean if show_p else qc_mean
    for i in range(n):
        cid = int(cluster_ids[i])
        col = cmap[cid]
        label = None
        if cid not in legended:
            label = "noise" if cid < 0 else f"cluster {cid}"
            legended.add(cid)
        ax1.plot(x, _smooth(left[:, i]), color=col, linewidth=1.4, alpha=0.8, label=label)
        ax2.plot(x, _smooth(qc_mean[:, i]), color=col, linestyle="-", linewidth=1.2, alpha=0.8)
        ax2.plot(x, _smooth(qd_mean[:, i]), color=col, linestyle="--", linewidth=1.0, alpha=0.6)

    if show_p:
        ax1.set_title("P(C) — cooperation probability — by cluster", fontsize=12)
        ax1.set_ylabel("P(C)")
        ax1.set_ylim(-0.02, 1.02)
    else:
        ax1.set_title("Q(C) by cluster", fontsize=12)
        ax1.set_ylabel("Q(C)")
    ax1.set_xlabel("Iterations")
    ax1.legend(loc="best", fontsize=9)

    ax2.set_title("Q(C) [solid] vs Q(D) [dashed]", fontsize=12)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Q-value")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(abs_path, dpi=dpi)
    plt.close()
    return abs_path

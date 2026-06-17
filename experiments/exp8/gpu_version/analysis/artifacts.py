"""
Persistent run artifacts — save once, re-plot forever.

Each experiment run writes two files into its output directory:

* ``artifacts.npz`` — a compressed numpy bundle with everything needed to
  redraw any figure or re-cluster the vertices **without re-running the
  simulation**:
    - ``adjacency``              (N, N) — the exact graph topology
    - ``degrees``                (N,)
    - ``qc_mean/qd_mean/p_mean`` (T_out, N) — replicate-averaged trajectories
    - ``qc_std/qd_std/p_std``    (T_out, N) — replicate std (for error bands)
    - ``cluster_labels``         (N,)
    - ``qc_final/qd_final``      (N,)  and ``conv_features`` (N, 3)
    - structural features        (N,) each: degree, clustering_coefficient,
      betweenness_centrality, eigenvector_centrality
    - ``record_every``           scalar
    - optionally the full ``(T_out, reps, N)`` histories when requested
* ``run_params.json`` — **all** launch parameters, down to the graph topology
  (simulation hyper-parameters, clustering settings, graph descriptor and the
  full edge list).

Use :func:`load_run_artifacts` to restore a run and :func:`replot_from_artifacts`
(or the ``replot_from_artifacts.py`` CLI) to regenerate the standard figures,
optionally re-clustering with different parameters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

ARTIFACT_VERSION = 1
ARTIFACTS_NPZ = "artifacts.npz"
RUN_PARAMS_JSON = "run_params.json"


@dataclass
class RunArtifacts:
    """In-memory view of a saved run (see :func:`load_run_artifacts`)."""

    adjacency: np.ndarray
    degrees: np.ndarray
    qc_mean: np.ndarray
    qd_mean: np.ndarray
    p_mean: np.ndarray
    qc_std: np.ndarray
    qd_std: np.ndarray
    p_std: np.ndarray
    cluster_labels: np.ndarray
    qc_final: np.ndarray
    qd_final: np.ndarray
    conv_features: np.ndarray
    topology_features: dict[str, np.ndarray]
    record_every: int
    run_params: dict
    # Present only if histories were saved in full.
    qc_hist_full: Optional[np.ndarray] = None
    qd_hist_full: Optional[np.ndarray] = None
    p_hist_full: Optional[np.ndarray] = None


def _as_cpu_numpy(adjacency) -> np.ndarray:
    """Coerce a torch tensor (possibly on CUDA/MPS) or array to a CPU ndarray.

    Using ``.detach().cpu().numpy()`` avoids both the ``can't convert cuda
    tensor to numpy`` error and the numpy>=2 ``copy`` ``FutureWarning`` that
    ``np.asarray(tensor)`` triggers.
    """
    if hasattr(adjacency, "detach"):
        return adjacency.detach().cpu().numpy()
    return np.asarray(adjacency)


def _edge_list(adjacency) -> list[list[int]]:
    adj = _as_cpu_numpy(adjacency)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    return [[int(i), int(j)] for i, j in zip(rows, cols)]


def build_run_params(
    *,
    topology_name: str,
    title: str,
    adjacency: np.ndarray,
    simulation_meta: dict,
    clustering_settings: dict,
    clustering_result: dict,
    graph_descriptor: Optional[dict] = None,
    n_final_steps: int,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the exhaustive ``run_params`` dict (incl. full topology).

    The ``graph_descriptor`` should describe how the graph was produced
    (e.g. ``{"family": "cubic", "n": 20}`` or ``{"family": "interpolated",
    "n": 20, "k": 2, "temperature": 0.5, "mode": "stochastic", "seed": 71}``).
    The full ``edge_list`` is always stored so the topology is fully recoverable.
    """
    adj = _as_cpu_numpy(adjacency)
    params = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "topology_name": topology_name,
        "title": title,
        "n_final_steps": int(n_final_steps),
        "graph": {
            **(graph_descriptor or {}),
            "num_nodes": int(adj.shape[0]),
            "num_edges": int((np.asarray(adj) > 0).sum() // 2),
            "adjacency_shape": list(adj.shape),
            "edge_list": _edge_list(adj),
        },
        "simulation": dict(simulation_meta),
        "clustering": {**dict(clustering_settings), **dict(clustering_result)},
    }
    if extra:
        params["extra"] = extra
    return params


def save_run_artifacts(
    out_dir: str,
    *,
    adjacency,
    degrees: np.ndarray,
    p_mean: np.ndarray,
    qc_mean: np.ndarray,
    qd_mean: np.ndarray,
    p_std: np.ndarray,
    qc_std: np.ndarray,
    qd_std: np.ndarray,
    cluster_labels: np.ndarray,
    qc_final: np.ndarray,
    qd_final: np.ndarray,
    conv_features: np.ndarray,
    topology_features: dict[str, np.ndarray],
    record_every: int,
    run_params: dict,
    p_hist_full: Optional[np.ndarray] = None,
    qc_hist_full: Optional[np.ndarray] = None,
    qd_hist_full: Optional[np.ndarray] = None,
) -> dict[str, str]:
    """Write ``artifacts.npz`` and ``run_params.json`` into ``out_dir``.

    Trajectories are stored as replicate mean/std ``(T_out, N)`` — compact,
    independent of batch size, and enough to redraw every figure in this
    project.  Pass the full ``(T_out, reps, N)`` arrays via ``*_hist_full`` to
    additionally persist the raw per-replicate histories.

    Returns:
        Dict with absolute paths of the two written files.
    """
    os.makedirs(out_dir, exist_ok=True)
    adj = adjacency.detach().cpu().numpy() if hasattr(adjacency, "detach") else np.asarray(adjacency)

    arrays = {
        "adjacency": adj.astype(np.float32),
        "degrees": np.asarray(degrees, dtype=np.float32),
        "qc_mean": np.asarray(qc_mean, dtype=np.float32),
        "qd_mean": np.asarray(qd_mean, dtype=np.float32),
        "p_mean": np.asarray(p_mean, dtype=np.float32),
        "qc_std": np.asarray(qc_std, dtype=np.float32),
        "qd_std": np.asarray(qd_std, dtype=np.float32),
        "p_std": np.asarray(p_std, dtype=np.float32),
        "cluster_labels": np.asarray(cluster_labels, dtype=np.int64),
        "qc_final": np.asarray(qc_final, dtype=np.float32),
        "qd_final": np.asarray(qd_final, dtype=np.float32),
        "conv_features": np.asarray(conv_features, dtype=np.float32),
        "record_every": np.int64(record_every),
    }
    for name, values in topology_features.items():
        arrays[f"topo__{name}"] = np.asarray(values, dtype=np.float32)

    if qc_hist_full is not None:
        arrays["qc_hist_full"] = np.asarray(qc_hist_full, dtype=np.float32)
        arrays["qd_hist_full"] = np.asarray(qd_hist_full, dtype=np.float32)
        arrays["p_hist_full"] = np.asarray(p_hist_full, dtype=np.float32)

    npz_path = os.path.abspath(os.path.join(out_dir, ARTIFACTS_NPZ))
    json_path = os.path.abspath(os.path.join(out_dir, RUN_PARAMS_JSON))
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w") as f:
        json.dump(run_params, f, indent=2)
    return {"artifacts": npz_path, "run_params": json_path}


def load_run_artifacts(path: str) -> RunArtifacts:
    """Load a run from a directory or a direct ``artifacts.npz`` path."""
    if os.path.isdir(path):
        npz_path = os.path.join(path, ARTIFACTS_NPZ)
        json_path = os.path.join(path, RUN_PARAMS_JSON)
    else:
        npz_path = path
        json_path = os.path.join(os.path.dirname(path), RUN_PARAMS_JSON)

    data = np.load(npz_path, allow_pickle=False)
    topo = {k[len("topo__"):]: data[k] for k in data.files if k.startswith("topo__")}

    run_params: dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            run_params = json.load(f)

    return RunArtifacts(
        adjacency=data["adjacency"],
        degrees=data["degrees"],
        qc_mean=data["qc_mean"],
        qd_mean=data["qd_mean"],
        p_mean=data["p_mean"],
        qc_std=data["qc_std"],
        qd_std=data["qd_std"],
        p_std=data["p_std"],
        cluster_labels=data["cluster_labels"],
        qc_final=data["qc_final"],
        qd_final=data["qd_final"],
        conv_features=data["conv_features"],
        topology_features=topo,
        record_every=int(data["record_every"]),
        run_params=run_params,
        qc_hist_full=data["qc_hist_full"] if "qc_hist_full" in data.files else None,
        qd_hist_full=data["qd_hist_full"] if "qd_hist_full" in data.files else None,
        p_hist_full=data["p_hist_full"] if "p_hist_full" in data.files else None,
    )


def replot_from_artifacts(
    path: str,
    *,
    out_dir: Optional[str] = None,
    layout: str = "circular",
    recluster: bool = False,
    cluster_method: str = "auto",
    n_final_steps: Optional[int] = None,
    dpi_curves: int = 130,
    dpi_graph: int = 150,
) -> dict[str, str]:
    """Regenerate the standard figures from saved artifacts (no simulation).

    Args:
        path: run directory or ``artifacts.npz`` path.
        out_dir: where to write the PNGs (defaults to the artifacts' directory).
        layout: NetworkX layout for the cluster graph.
        recluster: if ``True``, recompute convergence features (using the saved
            replicate-averaged histories) and re-cluster — useful for trying a
            different ``cluster_method`` or ``n_final_steps`` without re-running.
        cluster_method: clustering method used when ``recluster`` is ``True``.
        n_final_steps: averaging window when ``recluster`` is ``True``
            (defaults to the value stored in ``run_params``).

    Returns:
        Dict of written figure paths.
    """
    # Local imports keep numpy-only consumers free of matplotlib/networkx.
    from experiments.exp8.gpu_version.visualization.cluster_plotting import (
        plot_convergence_clusters, plot_q_curves_by_cluster,
    )

    art = load_run_artifacts(path)
    base = out_dir or (path if os.path.isdir(path) else os.path.dirname(path))
    os.makedirs(base, exist_ok=True)
    title = art.run_params.get("title", art.run_params.get("topology_name", "run"))

    labels = art.cluster_labels
    if recluster:
        from experiments.exp8.gpu_version.analysis.convergence_clustering import (
            compute_convergence_features, cluster_vertices_by_convergence,
        )
        window = n_final_steps if n_final_steps is not None else int(
            art.run_params.get("n_final_steps", 10_000))
        feats = compute_convergence_features(
            art.qc_mean, art.qd_mean,
            n_final_steps=window, record_every=art.record_every)
        # Reconstruct the Monte-Carlo noise scale from the saved std arrays so the
        # homogeneity guard behaves exactly as during the original run.
        reps_eff = max(1, int(art.run_params.get("simulation", {}).get("reps", 1)))
        k_rec = feats.n_records_used
        qc_se = art.qc_std[-k_rec:].mean(axis=0) / np.sqrt(reps_eff)
        qd_se = art.qd_std[-k_rec:].mean(axis=0) / np.sqrt(reps_eff)
        noise_scale = float(np.sqrt(qc_se ** 2 + qd_se ** 2).max())
        labels = cluster_vertices_by_convergence(
            feats.features, method=cluster_method, noise_scale=noise_scale).labels

    q_path = plot_q_curves_by_cluster(
        art.qc_mean, art.qd_mean, labels,
        os.path.join(base, "q_curves.png"),
        p_hist=art.p_mean, record_every=art.record_every,
        title=f"{title} | p(C) & Q by cluster", degrees=art.degrees, dpi=dpi_curves)
    g_path = plot_convergence_clusters(
        art.adjacency, labels, art.degrees,
        os.path.join(base, "convergence_clusters.png"),
        title=f"{title} | convergence clusters", layout=layout, dpi=dpi_graph)
    return {"q_curves": q_path, "convergence_clusters": g_path}

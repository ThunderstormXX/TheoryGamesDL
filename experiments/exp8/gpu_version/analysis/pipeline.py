"""
End-to-end per-topology analysis pipeline.

Glue that ties together simulation -> convergence features -> clustering ->
structural features -> artifacts, so the two runner scripts
(``run_all_convergence_topology_experiments.py`` and
``run_topology_phase_transition.py``) stay thin.

For one graph it produces, in ``out_dir``::

    q_curves.png              Q(C)/Q(D) trajectories coloured by cluster
    convergence_clusters.png  graph drawing coloured by cluster
    cluster_table.csv         per-vertex table (+ structural descriptors)
    summary.json              scalar summary + cluster/topology correlations

and returns the ``summary`` dict.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from experiments.exp8.gpu_version.analysis.simulation import run_convergence_simulation
from experiments.exp8.gpu_version.analysis.convergence_clustering import (
    compute_convergence_features,
    cluster_vertices_by_convergence,
    save_cluster_table,
    cluster_size_summary,
    largest_cluster_fraction,
)
from experiments.exp8.gpu_version.analysis.topology_features import (
    compute_topology_features,
    degree_distribution,
    TOPOLOGY_FEATURE_NAMES,
)
from experiments.exp8.gpu_version.analysis.artifacts import (
    build_run_params,
    save_run_artifacts,
)
from experiments.exp8.gpu_version.visualization.cluster_plotting import (
    plot_convergence_clusters,
    plot_q_curves_by_cluster,
)


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    """Correlation ratio (eta) between a categorical and a continuous variable.

    Measures how much of the variance in ``values`` is explained by the cluster
    assignment in ``categories``.  Returns ``eta in [0, 1]`` (``nan`` if undefined).
    This is the right association measure between *cluster id* (categorical) and a
    *structural feature* (continuous).
    """
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    categories, values = categories[mask], values[mask]
    if values.size < 2:
        return float("nan")
    total_var = values.var()
    if total_var <= 1e-12:
        return 0.0
    grand_mean = values.mean()
    ss_between = 0.0
    for cat in np.unique(categories):
        grp = values[categories == cat]
        ss_between += grp.size * (grp.mean() - grand_mean) ** 2
    ss_total = ((values - grand_mean) ** 2).sum()
    if ss_total <= 1e-12:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def _cluster_topology_correlations(
    cluster_ids: np.ndarray,
    topo: dict[str, np.ndarray],
) -> dict[str, float]:
    """eta(cluster, feature) for each structural feature (non-noise vertices)."""
    mask = cluster_ids >= 0
    out: dict[str, float] = {}
    for name in TOPOLOGY_FEATURE_NAMES:
        if name in topo:
            out[name] = correlation_ratio(cluster_ids[mask], topo[name][mask])
    return out


def _mean_q_per_cluster(
    cluster_ids: np.ndarray, qc_final: np.ndarray, qd_final: np.ndarray
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cid in np.unique(cluster_ids):
        key = "noise" if cid < 0 else str(int(cid))
        sel = cluster_ids == cid
        out[key] = {
            "mean_Q_C": float(qc_final[sel].mean()),
            "mean_Q_D": float(qd_final[sel].mean()),
            "size": int(sel.sum()),
        }
    return out


def analyze_from_sim(
    adjacency,
    sim,
    out_dir: str,
    *,
    topology_name: str,
    title: Optional[str] = None,
    seed: int = 42,
    n_final_steps: int = 10_000,
    cluster_method: str = "auto",
    cluster_scaling: str = "shared",
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 2,
    hdbscan_min_cluster_size: int = 2,
    save_artifacts: bool = True,
    save_data_artifacts: bool = True,
    save_full_histories: bool = False,
    layout: str = "circular",
    graph_descriptor: Optional[dict] = None,
    extra_summary: Optional[dict] = None,
    return_details: bool = False,
):
    """Run the convergence-cluster + topology analysis on an existing simulation.

    This is the post-simulation half of :func:`analyze_topology`, split out so the
    graph-fusion path can run one simulation on a block-diagonal super-graph and
    then analyse each original graph from a node-slice of that result
    (``sim.slice_nodes(...)``) without re-simulating.

    Args:
        adjacency: ``(N, N)`` adjacency of *this* graph (the block, when fused).
        sim: a :class:`analysis.simulation.SimulationResult` for this graph.
        seed: random_state for KMeans reproducibility.
        (other args mirror :func:`analyze_topology`).
    """
    title = title or topology_name
    record_every = sim.record_every
    if save_artifacts:
        os.makedirs(out_dir, exist_ok=True)

    # 2. convergence features (from replicate-mean trajectories)
    feats = compute_convergence_features(
        sim.qc_mean, sim.qd_mean,
        n_final_steps=n_final_steps, record_every=record_every)

    # Monte-Carlo standard error of the per-vertex final Q-values, used to tell
    # real convergence classes apart from finite-sample noise on homogeneous
    # (vertex-transitive) graphs.  Take the noisiest feature (Q_C - Q_D).
    k_rec = feats.n_records_used
    reps_eff = max(1, int(sim.meta.get("reps", 1)))
    qc_se = sim.qc_std[-k_rec:].mean(axis=0) / np.sqrt(reps_eff)
    qd_se = sim.qd_std[-k_rec:].mean(axis=0) / np.sqrt(reps_eff)
    noise_scale = float(np.sqrt(qc_se ** 2 + qd_se ** 2).max())

    # 3. clustering
    clust = cluster_vertices_by_convergence(
        feats.features, method=cluster_method, scaling=cluster_scaling,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        random_state=seed, noise_scale=noise_scale)

    # 4. structural features
    topo = compute_topology_features(adjacency)

    # 5/6/7. artifacts (figures, table, reusable data bundle)
    if save_artifacts:
        save_cluster_table(
            os.path.join(out_dir, "cluster_table.csv"),
            cluster_ids=clust.labels, degrees=sim.degrees,
            qc_final=feats.qc_final, qd_final=feats.qd_final,
            topology_features=topo)
        plot_q_curves_by_cluster(
            sim.qc_mean, sim.qd_mean, clust.labels,
            os.path.join(out_dir, "q_curves.png"),
            p_hist=sim.p_mean, record_every=record_every,
            title=f"{title} | p(C) & Q by cluster", degrees=sim.degrees)
        plot_convergence_clusters(
            adjacency, clust.labels, sim.degrees,
            os.path.join(out_dir, "convergence_clusters.png"),
            title=f"{title} | convergence clusters", layout=layout)

        if save_data_artifacts:
            run_params = build_run_params(
                topology_name=topology_name, title=title, adjacency=adjacency,
                simulation_meta=sim.meta,
                clustering_settings={
                    "method": cluster_method, "scaling": cluster_scaling,
                    "dbscan_eps": dbscan_eps, "dbscan_min_samples": dbscan_min_samples,
                    "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                    "random_state": seed, "noise_scale": noise_scale,
                },
                clustering_result={
                    "method_used": clust.method_used,
                    "n_clusters": int(clust.n_clusters),
                    "silhouette": clust.silhouette,
                    "chosen_params": clust.params,
                },
                graph_descriptor=graph_descriptor,
                n_final_steps=n_final_steps,
                extra=extra_summary,
            )
            save_run_artifacts(
                out_dir,
                adjacency=adjacency, degrees=sim.degrees,
                p_mean=sim.p_mean, qc_mean=sim.qc_mean, qd_mean=sim.qd_mean,
                p_std=sim.p_std, qc_std=sim.qc_std, qd_std=sim.qd_std,
                cluster_labels=clust.labels,
                qc_final=feats.qc_final, qd_final=feats.qd_final,
                conv_features=feats.features, topology_features=topo,
                record_every=record_every, run_params=run_params,
                p_hist_full=sim.p_hist if save_full_histories else None,
                qc_hist_full=sim.qc_hist if save_full_histories else None,
                qd_hist_full=sim.qd_hist if save_full_histories else None)

    # 8. summary + correlations
    summary = {
        "topology_name": topology_name,
        "title": title,
        "num_nodes": int(sim.degrees.shape[0]),
        "method_used": clust.method_used,
        "cluster_params": clust.params,
        "number_of_clusters": int(clust.n_clusters),
        "silhouette": clust.silhouette,
        "cluster_sizes": cluster_size_summary(clust.labels),
        "largest_cluster_fraction": largest_cluster_fraction(clust.labels),
        "mean_cooperation": float(sim.final_pc),
        "mean_Q_C": float(feats.qc_final.mean()),
        "mean_Q_D": float(feats.qd_final.mean()),
        "per_cluster_mean_Q": _mean_q_per_cluster(clust.labels, feats.qc_final, feats.qd_final),
        "degree_distribution": {str(k): v for k, v in degree_distribution(adjacency).items()},
        "cluster_topology_correlation_eta": _cluster_topology_correlations(clust.labels, topo),
        "n_final_records_averaged": int(feats.n_records_used),
        "simulation": sim.meta,
    }
    if extra_summary:
        summary.update(extra_summary)

    if save_artifacts:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if return_details:
        details = {
            "labels": clust.labels,
            "sim": sim,
            "features": feats,
            "cluster": clust,
            "topology_features": topo,
        }
        return summary, details
    return summary


def analyze_topology(
    adjacency,
    out_dir: str,
    *,
    topology_name: str,
    title: Optional[str] = None,
    # simulation parameters
    gamma: float = 0.0,
    beta: float = 1.0,
    learner_type: str = "q_learning",
    iters: int = 200_000,
    reps: int = 256,
    seed: int = 42,
    alpha: float = 0.01,
    record_every: int = 5_000,
    reward_type: str = "pp",
    b: float = 2.0,
    c: float = 1.0,
    bonus: float = 1.0,
    device=None,
    store_reps: str = "reduced",
    fast_reward: bool = True,
    progress: bool = False,
    progress_desc: str = "steps",
    progress_position: int = 1,
    # feature / clustering parameters
    n_final_steps: int = 10_000,
    cluster_method: str = "auto",
    cluster_scaling: str = "shared",
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 2,
    hdbscan_min_cluster_size: int = 2,
    # output control
    save_artifacts: bool = True,
    save_data_artifacts: bool = True,
    save_full_histories: bool = False,
    layout: str = "circular",
    graph_descriptor: Optional[dict] = None,
    extra_summary: Optional[dict] = None,
    return_details: bool = False,
):
    """Run the full convergence-cluster + topology analysis for one graph.

    Args:
        adjacency: ``(N, N)`` adjacency matrix.
        out_dir: directory to write artifacts into.
        topology_name: identifier stored in the summary.
        title: figure title (defaults to ``topology_name``).
        gamma..device: simulation parameters (see
            :func:`analysis.simulation.run_convergence_simulation`).
        n_final_steps: averaging window for the final Q-values.
        cluster_method: ``"auto" | "dbscan" | "hdbscan" | "kmeans"``.
        cluster_scaling: feature scaling mode (``"shared" | "standard" | "none"``).
        dbscan_eps, dbscan_min_samples, hdbscan_min_cluster_size: clustering params.
        save_artifacts: if ``False`` only computes and returns the summary
            (no PNG/CSV/JSON/NPZ written) — used by the phase-transition sweep for
            non-representative realizations.
        save_data_artifacts: also write ``artifacts.npz`` + ``run_params.json``
            (the reusable data bundle that lets figures be regenerated without
            re-simulating). Ignored when ``save_artifacts`` is ``False``.
        save_full_histories: store the full ``(T_out, reps, N)`` histories in
            the npz in addition to the replicate mean/std.
        layout: NetworkX layout for the cluster drawing.
        graph_descriptor: how the graph was produced (family/n/k/temperature/…);
            stored verbatim in ``run_params.json`` alongside the full edge list.
        extra_summary: extra key/values merged into the summary JSON.
        return_details: when ``True`` return ``(summary, details)`` where
            ``details`` holds the raw arrays (``labels``, ``sim``, ``features``,
            ``cluster``) for callers that want to render their own figures.

    Returns:
        The ``summary`` dict, or ``(summary, details)`` if ``return_details``.
    """
    title = title or topology_name

    # 1. simulate.  Full per-replicate histories are only produced by the
    # simulation when store_reps="full", so couple it to save_full_histories.
    effective_store_reps = "full" if save_full_histories else store_reps
    sim = run_convergence_simulation(
        adjacency, gamma=gamma, beta=beta, learner_type=learner_type,
        iters=iters, reps=reps, seed=seed, alpha=alpha, record_every=record_every,
        reward_type=reward_type, b=b, c=c, bonus=bonus, device=device,
        store_reps=effective_store_reps, fast_reward=fast_reward,
        progress=progress, progress_desc=progress_desc,
        progress_position=progress_position,
    )

    # 2-8. analyse the simulation result (shared with the fusion path).
    return analyze_from_sim(
        adjacency, sim, out_dir,
        topology_name=topology_name, title=title, seed=seed,
        n_final_steps=n_final_steps, cluster_method=cluster_method,
        cluster_scaling=cluster_scaling, dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        save_artifacts=save_artifacts, save_data_artifacts=save_data_artifacts,
        save_full_histories=save_full_histories, layout=layout,
        graph_descriptor=graph_descriptor, extra_summary=extra_summary,
        return_details=return_details,
    )

"""
Task 1 — vertex clustering by *convergence type*.

Motivation
----------
Vertices of the *same degree* can converge to different limiting Q-values
(e.g. degree-3 vertices splitting into two visually distinct Q(C) levels).
Degree alone is therefore not enough to describe asymptotic behaviour.  This
module discovers those convergence classes automatically.

Pipeline
--------
1. :func:`compute_convergence_features` — for every vertex build the feature
   vector ``[Q_C_final, Q_D_final, Q_C_final - Q_D_final]`` using the averaged
   final values over the last ``N`` learning steps and over all replicates.
2. :func:`cluster_vertices_by_convergence` — standardize the features and run an
   automatic clustering algorithm.  Supported methods (in the task's preference
   order): ``dbscan``, ``hdbscan``, ``kmeans`` (with silhouette-based model
   selection).  ``method="auto"`` tries them in that order with sensible
   fallbacks.
3. :func:`save_cluster_table` — persist a tidy per-vertex CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

try:  # sklearn >= 1.3 ships HDBSCAN; the standalone `hdbscan` package is optional
    from sklearn.cluster import HDBSCAN as _SKHDBSCAN
except ImportError:  # pragma: no cover
    _SKHDBSCAN = None

try:
    import hdbscan as _hdbscan_pkg  # type: ignore
except ImportError:  # pragma: no cover
    _hdbscan_pkg = None

ClusterMethod = Literal["auto", "dbscan", "hdbscan", "kmeans"]

# Canonical ordering of the convergence feature columns.
CONVERGENCE_FEATURE_NAMES: tuple[str, ...] = ("Q_C_final", "Q_D_final", "Q_C_minus_Q_D")


@dataclass
class ConvergenceFeatures:
    """Per-vertex convergence features.

    Attributes:
        features: ``(N, 3)`` array ``[Q_C_final, Q_D_final, Q_C_final-Q_D_final]``.
        qc_final: ``(N,)`` averaged final Q(C).
        qd_final: ``(N,)`` averaged final Q(D).
        n_records_used: number of trailing snapshots averaged.
    """

    features: np.ndarray
    qc_final: np.ndarray
    qd_final: np.ndarray
    n_records_used: int


@dataclass
class ClusterResult:
    """Result of :func:`cluster_vertices_by_convergence`.

    Attributes:
        labels: ``(N,)`` integer cluster id per vertex; ``-1`` marks noise
            (density methods only).
        method_used: the algorithm actually applied.
        n_clusters: number of clusters excluding noise.
        silhouette: silhouette score of the final labelling (``nan`` if undefined).
        params: parameters of the chosen algorithm.
    """

    labels: np.ndarray
    method_used: str
    n_clusters: int
    silhouette: float
    params: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — features
# ════════════════════════════════════════════════════════════════════════════

def compute_convergence_features(
    qc_hist: np.ndarray,
    qd_hist: np.ndarray,
    *,
    n_final_steps: int = 10_000,
    record_every: int = 5_000,
) -> ConvergenceFeatures:
    """Build per-vertex convergence features from Q-value histories.

    The histories are recorded every ``record_every`` learning steps, so the
    last ``N = n_final_steps`` *steps* correspond to
    ``max(1, n_final_steps // record_every)`` trailing snapshots.  Each feature
    is averaged over those snapshots **and** over all replicates.

    Accepts either the full ``(T_out, reps, N)`` histories from a simulation, or
    the already replicate-averaged ``(T_out, N)`` histories restored from saved
    artifacts (see :mod:`analysis.artifacts`) — so vertices can be re-clustered
    for a different ``n_final_steps`` window without re-running the simulation.

    Args:
        qc_hist: ``(T_out, reps, N)`` or ``(T_out, N)`` history of Q(C).
        qd_hist: same shape as ``qc_hist`` — history of Q(D).
        n_final_steps: window length, in learning steps, to average over.
        record_every: snapshot stride used when the histories were recorded.

    Returns:
        A :class:`ConvergenceFeatures`.
    """
    if qc_hist.shape != qd_hist.shape:
        raise ValueError(
            f"qc_hist and qd_hist must share a shape, got {qc_hist.shape} vs {qd_hist.shape}")
    if qc_hist.ndim not in (2, 3):
        raise ValueError(
            f"expected (T_out, reps, N) or (T_out, N) histories, got {qc_hist.shape}")

    t_out = qc_hist.shape[0]
    n_records = max(1, min(t_out, n_final_steps // max(1, record_every)))

    # Average over the trailing records and (if present) the replicate axis.
    reduce_axes = (0, 1) if qc_hist.ndim == 3 else (0,)
    qc_final = qc_hist[-n_records:].mean(axis=reduce_axes)  # (N,)
    qd_final = qd_hist[-n_records:].mean(axis=reduce_axes)  # (N,)
    diff = qc_final - qd_final

    features = np.stack([qc_final, qd_final, diff], axis=1).astype(np.float64)
    return ConvergenceFeatures(
        features=features,
        qc_final=qc_final,
        qd_final=qd_final,
        n_records_used=n_records,
    )


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — clustering
# ════════════════════════════════════════════════════════════════════════════

ScalingMode = Literal["shared", "standard", "none"]


def _scale_features(x: np.ndarray, mode: ScalingMode) -> np.ndarray:
    """Scale convergence features prior to clustering.

    The three features (``Q_C_final``, ``Q_D_final``, their difference) are
    **already in the same Q-value units**, so per-column z-scoring is harmful:
    it amplifies a near-constant feature (e.g. Q(D), which often converges to the
    same value on every vertex) into pure noise and over-segments the clusters.

    Modes:
        ``"shared"``  (default): divide the whole matrix by a single robust
            scale — the largest per-column standard deviation.  Relative scales
            between features are preserved, flat features stay flat, and the
            DBSCAN ``eps`` becomes a dimensionless fraction of the dominant
            feature's spread.
        ``"standard"``: classic per-column z-score (use only when you know the
            features are on different scales).
        ``"none"``: raw features.
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "none":
        return x
    if mode == "standard":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-12] = 1.0
        return (x - mean) / std
    if mode == "shared":
        scale = float(np.max(x.std(axis=0)))
        if scale < 1e-12:
            scale = 1.0
        return (x - x.mean(axis=0)) / scale
    raise ValueError(f"Unknown scaling mode: {mode!r}")


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score that degrades gracefully on degenerate labellings."""
    mask = labels >= 0  # ignore noise points
    uniq = np.unique(labels[mask])
    if uniq.size < 2 or mask.sum() <= uniq.size:
        return float("nan")
    try:
        return float(silhouette_score(x[mask], labels[mask]))
    except Exception:
        return float("nan")


def _count_clusters(labels: np.ndarray) -> int:
    """Number of distinct non-noise cluster ids."""
    return int(np.unique(labels[labels >= 0]).size)


def _kmeans_with_silhouette(
    x: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> ClusterResult:
    """KMeans with automatic ``k`` selection by silhouette score.

    Falls back to a single cluster when fewer than two samples are available or
    no ``k`` yields a valid silhouette.
    """
    n = x.shape[0]
    k_hi = min(k_max, n - 1)
    if n < 3 or k_hi < 2:
        return ClusterResult(
            labels=np.zeros(n, dtype=int), method_used="kmeans",
            n_clusters=1, silhouette=float("nan"),
            params={"k": 1, "reason": "too_few_samples"})

    best: Optional[ClusterResult] = None
    for k in range(max(2, k_min), k_hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(x)
        score = _safe_silhouette(x, labels)
        if np.isnan(score):
            continue
        if best is None or score > best.silhouette:
            best = ClusterResult(
                labels=labels, method_used="kmeans", n_clusters=k,
                silhouette=score, params={"k": k, "random_state": random_state})

    if best is None:  # no valid silhouette anywhere -> one cluster
        return ClusterResult(
            labels=np.zeros(n, dtype=int), method_used="kmeans",
            n_clusters=1, silhouette=float("nan"),
            params={"k": 1, "reason": "no_valid_silhouette"})
    return best


def _run_dbscan(x: np.ndarray, *, eps: float, min_samples: int) -> ClusterResult:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    return ClusterResult(
        labels=labels, method_used="dbscan", n_clusters=_count_clusters(labels),
        silhouette=_safe_silhouette(x, labels),
        params={"eps": eps, "min_samples": min_samples})


def _run_hdbscan(x: np.ndarray, *, min_cluster_size: int) -> Optional[ClusterResult]:
    if _SKHDBSCAN is not None:
        model = _SKHDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(x)
        backend = "sklearn"
    elif _hdbscan_pkg is not None:
        model = _hdbscan_pkg.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(x)
        backend = "hdbscan_pkg"
    else:
        return None
    return ClusterResult(
        labels=labels, method_used="hdbscan", n_clusters=_count_clusters(labels),
        silhouette=_safe_silhouette(x, labels),
        params={"min_cluster_size": min_cluster_size, "backend": backend})


def is_homogeneous(
    features: np.ndarray,
    *,
    noise_scale: Optional[float] = None,
    sigma: float = 3.0,
    rel_tol: float = 0.03,
    abs_tol: float = 1e-4,
) -> bool:
    """Decide whether all vertices share a single convergence class.

    On a vertex-transitive graph (Ring, circulants, …) every vertex is
    structurally identical, so the only across-vertex variation in the final
    Q-values is finite-sample Monte-Carlo noise.  Feeding that to DBSCAN with the
    ``"shared"`` scaling would amplify the noise to unit scale and split it into
    spurious clusters.  This guard catches the case first.

    The across-vertex spread (max per-column std) is compared to a noise scale:

    * If ``noise_scale`` is given (the Monte-Carlo standard error of the
      per-vertex mean, which the pipeline derives from the replicate std and the
      batch size), the data is homogeneous when the spread is within
      ``sigma`` standard errors — i.e. the variation is not statistically
      significant.
    * Otherwise a relative fallback is used: spread below ``rel_tol`` of the
      Q-value magnitude (plus ``abs_tol``) counts as homogeneous.
    """
    x = np.asarray(features, dtype=np.float64)
    spread = float(np.max(x.std(axis=0))) if x.size else 0.0
    if noise_scale is not None and noise_scale > 0:
        return spread <= sigma * float(noise_scale)
    magnitude = float(np.max(np.abs(x))) if x.size else 0.0
    return spread <= rel_tol * magnitude + abs_tol


def cluster_vertices_by_convergence(
    features: np.ndarray,
    *,
    method: ClusterMethod = "auto",
    scaling: ScalingMode = "shared",
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 2,
    hdbscan_min_cluster_size: int = 2,
    kmeans_k_min: int = 2,
    kmeans_k_max: int = 10,
    random_state: int = 42,
    noise_scale: Optional[float] = None,
    homogeneity_sigma: float = 3.0,
) -> ClusterResult:
    """Cluster vertices by their convergence features.

    Features are scaled with :func:`_scale_features` (``"shared"`` by default,
    which divides the whole matrix by the dominant feature's spread).  Because
    ``Q_C_final``, ``Q_D_final`` and their difference are already in the same
    Q-value units, this is preferred over per-column standardization.

    ``method="auto"`` follows the task's preference order and is robust to
    degenerate outputs:

    1. **DBSCAN** — accepted if it finds >= 2 clusters.
    2. **HDBSCAN** — tried next if available and DBSCAN was degenerate.
    3. **KMeans + silhouette** — final fallback (always returns a labelling).

    Args:
        features: ``(N, d)`` convergence feature matrix.
        method: clustering algorithm or ``"auto"``.
        scaling: feature scaling mode (``"shared" | "standard" | "none"``).
        dbscan_eps, dbscan_min_samples: DBSCAN parameters (on scaled data; with
            ``"shared"`` scaling ``eps`` is a fraction of the dominant spread).
        hdbscan_min_cluster_size: HDBSCAN parameter.
        kmeans_k_min, kmeans_k_max: KMeans search range.
        random_state: seed for KMeans reproducibility.
        noise_scale: Monte-Carlo standard error of the per-vertex final Q (passed
            by the pipeline). When the across-vertex spread is within
            ``homogeneity_sigma`` of it, all vertices are deemed one class — this
            prevents spurious clusters on vertex-transitive graphs.
        homogeneity_sigma: significance threshold for the homogeneity guard.

    Returns:
        A :class:`ClusterResult`.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D (N, d), got shape {x.shape}")
    n = x.shape[0]

    if n <= 1:
        return ClusterResult(
            labels=np.zeros(n, dtype=int), method_used="trivial",
            n_clusters=int(n), silhouette=float("nan"),
            params={"reason": "n<=1"})

    # Guard: don't cluster pure Monte-Carlo noise on (near-)homogeneous graphs.
    if is_homogeneous(x, noise_scale=noise_scale, sigma=homogeneity_sigma):
        return ClusterResult(
            labels=np.zeros(n, dtype=int), method_used="homogeneous",
            n_clusters=1, silhouette=float("nan"),
            params={"reason": "spread_within_noise",
                    "noise_scale": (None if noise_scale is None else float(noise_scale)),
                    "spread": float(np.max(x.std(axis=0)))})

    x_proc = _scale_features(x, scaling)

    if method == "dbscan":
        return _run_dbscan(x_proc, eps=dbscan_eps, min_samples=dbscan_min_samples)
    if method == "hdbscan":
        res = _run_hdbscan(x_proc, min_cluster_size=hdbscan_min_cluster_size)
        if res is None:
            raise RuntimeError("HDBSCAN backend not available (install scikit-learn>=1.3 or hdbscan)")
        return res
    if method == "kmeans":
        return _kmeans_with_silhouette(
            x_proc, k_min=kmeans_k_min, k_max=kmeans_k_max, random_state=random_state)
    if method != "auto":
        raise ValueError(f"Unknown clustering method: {method!r}")

    # ── auto: DBSCAN -> HDBSCAN -> KMeans ──
    dbscan_res = _run_dbscan(x_proc, eps=dbscan_eps, min_samples=dbscan_min_samples)
    if dbscan_res.n_clusters >= 2:
        return dbscan_res

    hdbscan_res = _run_hdbscan(x_proc, min_cluster_size=hdbscan_min_cluster_size)
    if hdbscan_res is not None and hdbscan_res.n_clusters >= 2:
        return hdbscan_res

    return _kmeans_with_silhouette(
        x_proc, k_min=kmeans_k_min, k_max=kmeans_k_max, random_state=random_state)


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — persistence
# ════════════════════════════════════════════════════════════════════════════

def build_cluster_table(
    *,
    cluster_ids: np.ndarray,
    degrees: np.ndarray,
    qc_final: np.ndarray,
    qd_final: np.ndarray,
    topology_features: Optional[dict[str, np.ndarray]] = None,
):
    """Assemble the per-vertex results table as a :class:`pandas.DataFrame`.

    Columns: ``vertex_id, degree, cluster_id, Q_C_final, Q_D_final`` plus any
    structural descriptors supplied in ``topology_features`` (so convergence
    class can later be correlated with structural position).
    """
    import pandas as pd

    n = len(cluster_ids)
    data = {
        "vertex_id": np.arange(n, dtype=int),
        "degree": np.asarray(degrees).round().astype(int),
        "cluster_id": np.asarray(cluster_ids).astype(int),
        "Q_C_final": np.asarray(qc_final, dtype=float),
        "Q_D_final": np.asarray(qd_final, dtype=float),
    }
    if topology_features:
        for name, values in topology_features.items():
            if name == "degree":  # already present as integer column
                continue
            data[name] = np.asarray(values, dtype=float)
    return pd.DataFrame(data)


def save_cluster_table(
    path: str,
    *,
    cluster_ids: np.ndarray,
    degrees: np.ndarray,
    qc_final: np.ndarray,
    qd_final: np.ndarray,
    topology_features: Optional[dict[str, np.ndarray]] = None,
):
    """Write the per-vertex table to ``path`` (CSV) and return the DataFrame."""
    import os

    df = build_cluster_table(
        cluster_ids=cluster_ids, degrees=degrees,
        qc_final=qc_final, qd_final=qd_final,
        topology_features=topology_features)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def cluster_size_summary(labels: np.ndarray) -> dict[str, int]:
    """``{str(cluster_id): size}`` including a ``"noise"`` bucket for -1."""
    out: dict[str, int] = {}
    for lbl in np.unique(labels):
        key = "noise" if lbl < 0 else str(int(lbl))
        out[key] = int(np.sum(labels == lbl))
    return out


def largest_cluster_fraction(labels: np.ndarray) -> float:
    """Fraction of vertices in the largest *non-noise* cluster."""
    non_noise = labels[labels >= 0]
    if non_noise.size == 0:
        return 0.0
    _, counts = np.unique(non_noise, return_counts=True)
    return float(counts.max() / labels.size)

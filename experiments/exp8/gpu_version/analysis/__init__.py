"""
Analysis subpackage for vertex-convergence and topology studies.

This package is *additive*: it reuses the existing learning core
(:class:`BatchedGPUQLearner`, :class:`BatchedGPUSARSALearner`) and graph
generators without modifying them.  It provides:

* :mod:`analysis.simulation`            — a seedable simulation wrapper that
  returns per-vertex Q-value / P(C) histories.
* :mod:`analysis.convergence_clustering`— vertex convergence features and the
  automatic clustering pipeline (DBSCAN / HDBSCAN / KMeans+silhouette).
* :mod:`analysis.topology_features`     — structural per-vertex descriptors
  (degree, clustering coefficient, betweenness / eigenvector centrality).
* :mod:`analysis.interpolation`         — the continuous ``temperature`` family
  of graphs interpolating between a k-regular and a (k+1)-regular graph.
* :mod:`analysis.artifacts`             — save/load the reusable per-run data
  bundle (``artifacts.npz`` + ``run_params.json``) and regenerate figures from
  it without re-simulating.

See ``run_all_convergence_topology_experiments.py`` and
``run_topology_phase_transition.py`` for end-to-end usage.
"""

from . import (  # noqa: F401
    simulation, convergence_clustering, topology_features, interpolation, artifacts,
)

__all__ = [
    "simulation",
    "convergence_clustering",
    "topology_features",
    "interpolation",
    "artifacts",
]

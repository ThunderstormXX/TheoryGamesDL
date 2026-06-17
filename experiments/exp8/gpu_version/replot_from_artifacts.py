"""
Regenerate figures from saved run artifacts — no simulation required.

Every run produced by ``run_all_convergence_topology_experiments.py`` /
``run_topology_phase_transition.py`` stores an ``artifacts.npz`` +
``run_params.json`` bundle (see :mod:`analysis.artifacts`).  This CLI reloads
those and redraws ``q_curves.png`` and ``convergence_clusters.png``, optionally
re-clustering with different settings or restyling the graph layout.

Examples
--------
Redraw one run::

    python -m experiments.exp8.gpu_version.replot_from_artifacts \\
        results/convergence_topology/cubic_n20_g0.9_b1.0_q_learning

Recurse over every run under a directory::

    python -m experiments.exp8.gpu_version.replot_from_artifacts \\
        results/convergence_topology --recursive

Re-cluster with KMeans and a shorter averaging window, spring layout::

    python -m experiments.exp8.gpu_version.replot_from_artifacts <run_dir> \\
        --recluster --cluster-method kmeans --n-final-steps 5000 --layout spring
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from experiments.exp8.gpu_version.analysis.artifacts import (  # noqa: E402
    ARTIFACTS_NPZ, replot_from_artifacts,
)


def find_run_dirs(root: str, recursive: bool) -> list[str]:
    """Locate run directories (those containing ``artifacts.npz``)."""
    if os.path.isfile(root):  # a direct .npz path
        return [root]
    if os.path.exists(os.path.join(root, ARTIFACTS_NPZ)):
        return [root]
    if not recursive:
        return []
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if ARTIFACTS_NPZ in files:
            found.append(dirpath)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="run directory, artifacts.npz, or a parent dir")
    parser.add_argument("--recursive", action="store_true",
                        help="recurse and replot every run found under PATH")
    parser.add_argument("--out-dir", default=None,
                        help="write figures here instead of next to the artifacts")
    parser.add_argument("--layout", default="circular",
                        choices=["circular", "spring", "kamada_kawai"])
    parser.add_argument("--recluster", action="store_true",
                        help="recompute clusters from saved histories before plotting")
    parser.add_argument("--cluster-method", default="auto",
                        choices=["auto", "dbscan", "hdbscan", "kmeans"])
    parser.add_argument("--n-final-steps", type=int, default=None,
                        help="averaging window for --recluster (default: from run_params)")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.path, args.recursive)
    if not run_dirs:
        print(f"No artifacts.npz found under {args.path!r} "
              f"(use --recursive to search subdirectories).")
        return

    print(f"Replotting {len(run_dirs)} run(s)...")
    for rd in run_dirs:
        out = replot_from_artifacts(
            rd, out_dir=args.out_dir, layout=args.layout,
            recluster=args.recluster, cluster_method=args.cluster_method,
            n_final_steps=args.n_final_steps)
        print(f"  {rd} -> {os.path.basename(out['q_curves'])}, "
              f"{os.path.basename(out['convergence_clusters'])}")


if __name__ == "__main__":
    main()

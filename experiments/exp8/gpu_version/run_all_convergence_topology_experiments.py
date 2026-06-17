"""
Task 2 — mass run of convergence-cluster + topology analysis.

Sweeps every existing k-regular graph family, every project graph size, and a
representative set of supervisor scenarios (gamma / beta / learner).  For each
combination it writes::

    results/convergence_topology/<topology_name>/
        q_curves.png
        convergence_clusters.png
        cluster_table.csv
        summary.json

and an aggregate ``index_summary.json`` at the top level.  The per-topology
layout mirrors ``supervisor_results`` (a folder per run + a ``summary.json``).

Examples
--------
Quick smoke run (tiny, finishes in seconds, CPU-friendly)::

    python -m experiments.exp8.gpu_version.run_all_convergence_topology_experiments --smoke

Default run::

    python -m experiments.exp8.gpu_version.run_all_convergence_topology_experiments

Custom subset::

    python -m experiments.exp8.gpu_version.run_all_convergence_topology_experiments \\
        --graphs cubic mixed23 --sizes 10 20 --iters 300000 --reps 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils  # noqa: E402

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
gpu_utils.gpu_config.device = DEVICE

from experiments.exp8.gpu_version.core.graph_structure import (  # noqa: E402
    RingGraph, CubicCirculantGraph, QuarticCirculantGraph,
    QuinticCirculantGraph, Mixed23Graph, Mixed34Graph,
)
from experiments.exp8.gpu_version.analysis.pipeline import analyze_topology  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ────────────────────────────────────────────────────────────────────────────
# Graph registry: key -> (class, nice name, size-constraint predicate)
# ────────────────────────────────────────────────────────────────────────────
GRAPH_REGISTRY = {
    "ring":    (RingGraph,            "Ring (2-reg)",   lambda n: n >= 3),
    "cubic":   (CubicCirculantGraph,  "Cubic (3-reg)",  lambda n: n % 2 == 0 and n >= 4),
    "quartic": (QuarticCirculantGraph, "Quartic (4-reg)", lambda n: n >= 5),
    "quintic": (QuinticCirculantGraph, "Quintic (5-reg)", lambda n: n % 2 == 0 and n >= 6),
    "mixed23": (Mixed23Graph,         "Mixed (2/3)",    lambda n: n >= 4),
    "mixed34": (Mixed34Graph,         "Mixed (3/4)",    lambda n: n % 2 == 0 and n >= 6),
}

# Representative supervisor scenarios: (gamma, beta, learner).
# gamma=0.9 is where the multi-cluster Q separation is most visible.
DEFAULT_SCENARIOS = [
    (0.0, 1.0, "q_learning"),
    (0.9, 1.0, "q_learning"),
]

DEFAULT_SIZES = [10, 20, 50]


def build_scenarios(args) -> list[tuple[float, float, str]]:
    if args.gammas or args.betas or args.learners:
        gammas = args.gammas or [0.0, 0.9]
        betas = args.betas or [1.0]
        learners = args.learners or ["q_learning"]
        return [(g, b, l) for g in gammas for b in betas for l in learners]
    return list(DEFAULT_SCENARIOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPH_REGISTRY),
                        choices=list(GRAPH_REGISTRY), help="graph families to run")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="graph sizes (n)")
    parser.add_argument("--gammas", nargs="+", type=float, default=None)
    parser.add_argument("--betas", nargs="+", type=float, default=None)
    parser.add_argument("--learners", nargs="+", default=None,
                        choices=["q_learning", "sarsa"])
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--reps", type=int, default=256)
    parser.add_argument("--record-every", type=int, default=5_000)
    parser.add_argument("--n-final-steps", type=int, default=10_000)
    parser.add_argument("--cluster-method", default="auto",
                        choices=["auto", "dbscan", "hdbscan", "kmeans"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-full-histories", action="store_true",
                        help="also store full (T_out, reps, N) histories in artifacts.npz")
    parser.add_argument("--no-data-artifacts", action="store_true",
                        help="skip artifacts.npz / run_params.json (figures only)")
    parser.add_argument("--out-dir", default=None,
                        help="base output dir (default: ./results/convergence_topology)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny/fast config for a correctness check")
    args = parser.parse_args()

    if args.smoke:
        args.sizes = [10]
        args.iters = 3_000
        args.reps = 16
        args.record_every = 1_000
        args.n_final_steps = 2_000

    base_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", "convergence_topology")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    scenarios = build_scenarios(args)

    # Enumerate valid (graph, n) pairs.
    runs: list[tuple[str, int, float, float, str]] = []
    for gkey in args.graphs:
        _, _, valid = GRAPH_REGISTRY[gkey]
        for n in args.sizes:
            if not valid(n):
                print(f"  [skip] {gkey} n={n} (size constraint)")
                continue
            for gamma, beta, learner in scenarios:
                runs.append((gkey, n, gamma, beta, learner))

    print("=" * 64)
    print(f"  Mass convergence-topology run on device: {DEVICE}")
    print(f"  graphs={args.graphs} sizes={args.sizes}")
    print(f"  scenarios={scenarios}")
    print(f"  iters={args.iters} reps={args.reps} -> {len(runs)} runs")
    print(f"  output: {base_dir}")
    print("=" * 64)

    index: dict[str, dict] = {}
    t0 = time.time()
    for gkey, n, gamma, beta, learner in tqdm(runs, desc="topologies"):
        graph_cls, nice_name, _ = GRAPH_REGISTRY[gkey]
        topology_name = f"{gkey}_n{n}_g{gamma}_b{beta}_{learner}"
        out_dir = os.path.join(base_dir, topology_name)
        title = f"{nice_name} | n={n}, γ={gamma}, β={beta}, {learner}"
        try:
            adj = graph_cls(n, DEVICE).generate_adjacency_matrix()
            summary = analyze_topology(
                adj, out_dir,
                topology_name=topology_name, title=title,
                gamma=gamma, beta=beta, learner_type=learner,
                iters=args.iters, reps=args.reps, seed=args.seed,
                record_every=args.record_every, n_final_steps=args.n_final_steps,
                cluster_method=args.cluster_method, device=DEVICE,
                save_data_artifacts=not args.no_data_artifacts,
                save_full_histories=args.save_full_histories,
                graph_descriptor={"family": gkey, "graph_class": graph_cls.__name__,
                                  "n": n},
                extra_summary={"graph_family": gkey, "n": n,
                               "gamma": gamma, "beta": beta, "learner": learner},
            )
            index[topology_name] = {
                "number_of_clusters": summary["number_of_clusters"],
                "largest_cluster_fraction": summary["largest_cluster_fraction"],
                "mean_cooperation": summary["mean_cooperation"],
                "method_used": summary["method_used"],
                "cluster_topology_correlation_eta": summary["cluster_topology_correlation_eta"],
            }
        except Exception as exc:  # keep going across the sweep
            print(f"  [error] {topology_name}: {exc}")
            traceback.print_exc()
            index[topology_name] = {"error": str(exc)}

    with open(os.path.join(base_dir, "index_summary.json"), "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {len(runs)} runs in {elapsed:.0f}s. Index: "
          f"{os.path.join(base_dir, 'index_summary.json')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"Fatal error: {e}")
        traceback.print_exc()

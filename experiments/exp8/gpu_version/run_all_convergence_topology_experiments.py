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
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Mixed45Graph, Mixed56Graph,
)
from experiments.exp8.gpu_version.analysis.pipeline import (  # noqa: E402
    analyze_topology, analyze_from_sim,
)
from experiments.exp8.gpu_version.analysis.simulation import (  # noqa: E402
    suggest_reps, run_convergence_simulation,
)
from experiments.exp8.gpu_version.analysis.fusion import block_diagonal_adjacency  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x=None, **kwargs):
        return x if x is not None else iter(())


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
    "mixed45": (Mixed45Graph,         "Mixed (4/5)",    lambda n: n % 2 == 0 and n >= 6),
    "mixed56": (Mixed56Graph,         "Mixed (5/6)",    lambda n: n % 2 == 0 and n >= 8),
}

# Richer default sweep (gamma=0.9/0.95 are where multi-cluster Q separation
# is most visible; betas span Boltzmann temperatures).
DEFAULT_GAMMAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
DEFAULT_BETAS = [0.5, 1.0, 2.0, 4.0]
DEFAULT_LEARNERS = ["q_learning"]
DEFAULT_SIZES = [10, 20, 50, 100]


def build_scenarios(args) -> list[tuple[float, float, str]]:
    """Cartesian product of gamma x beta x learner (rich defaults if unset)."""
    gammas = args.gammas or DEFAULT_GAMMAS
    betas = args.betas or DEFAULT_BETAS
    learners = args.learners or DEFAULT_LEARNERS
    return [(g, b, l) for g in gammas for b in betas for l in learners]


def _resolve_reps(n: int, args, *, n_workers: int) -> int:
    """Resolve the batch size for one graph size, honouring --auto-reps.

    Under ``--auto-reps`` the per-run batch targets a slice of free VRAM; when
    several workers share one GPU the budget is divided by ``n_workers`` so the
    processes don't collectively over-commit memory.
    """
    if not getattr(args, "auto_reps", False):
        return args.reps
    return suggest_reps(
        n, device=DEVICE,
        vram_fraction=args.vram_fraction / max(1, n_workers),
        max_reps=args.max_reps)


def _run_one_config(payload: dict) -> tuple[str, dict]:
    """Run a single (graph, n, gamma, beta, learner) config; return index entry.

    Defined at module scope so it is picklable by ``ProcessPoolExecutor``.  Each
    worker rebuilds its own graph on ``DEVICE`` (tensors are never pickled) and
    writes its own output directory, so runs never contend.
    """
    gkey = payload["gkey"]
    n = payload["n"]
    gamma = payload["gamma"]
    beta = payload["beta"]
    learner = payload["learner"]
    graph_cls, nice_name, _ = GRAPH_REGISTRY[gkey]
    topology_name = f"{gkey}_n{n}_g{gamma}_b{beta}_{learner}"
    out_dir = os.path.join(payload["base_dir"], topology_name)
    title = f"{nice_name} | n={n}, γ={gamma}, β={beta}, {learner}"
    try:
        adj = graph_cls(n, DEVICE).generate_adjacency_matrix()
        summary = analyze_topology(
            adj, out_dir,
            topology_name=topology_name, title=title,
            gamma=gamma, beta=beta, learner_type=learner,
            iters=payload["iters"], reps=payload["reps"], seed=payload["seed"],
            record_every=payload["record_every"], n_final_steps=payload["n_final_steps"],
            cluster_method=payload["cluster_method"], device=DEVICE,
            store_reps=payload["store_reps"],
            save_data_artifacts=payload["save_data_artifacts"],
            save_full_histories=payload["save_full_histories"],
            progress=payload["progress"], progress_desc=topology_name,
            graph_descriptor={"family": gkey, "graph_class": graph_cls.__name__, "n": n},
            extra_summary={"graph_family": gkey, "n": n,
                           "gamma": gamma, "beta": beta, "learner": learner},
        )
        return topology_name, {
            "number_of_clusters": summary["number_of_clusters"],
            "largest_cluster_fraction": summary["largest_cluster_fraction"],
            "mean_cooperation": summary["mean_cooperation"],
            "method_used": summary["method_used"],
            "reps": payload["reps"],
            "cluster_topology_correlation_eta": summary["cluster_topology_correlation_eta"],
        }
    except Exception as exc:
        traceback.print_exc()
        return topology_name, {"error": str(exc)}


def _run_fused_scenario(payload: dict) -> list[tuple[str, dict]]:
    """Run ALL graphs of one (gamma, beta, learner) scenario in a single sim.

    Builds a block-diagonal super-graph from every selected ``(graph, n)``, runs
    one ``iters``-step simulation on it (filling the GPU and paying the Python
    loop overhead once instead of per graph), then analyses each graph from its
    node-slice.  Module scope so it is picklable by ``ProcessPoolExecutor``.
    """
    gamma, beta, learner = payload["gamma"], payload["beta"], payload["learner"]
    graphs = payload["graphs"]  # list of (gkey, n)
    base_dir = payload["base_dir"]
    scen_tag = f"g{gamma}_b{beta}_{learner}"

    try:
        adjs, blocks = [], []
        for gkey, n in graphs:
            graph_cls, nice_name, _ = GRAPH_REGISTRY[gkey]
            adjs.append(graph_cls(n, DEVICE).generate_adjacency_matrix())
            blocks.append((gkey, n, nice_name))

        big_adj, ranges = block_diagonal_adjacency(adjs, device=DEVICE)

        sim = run_convergence_simulation(
            big_adj, gamma=gamma, beta=beta, learner_type=learner,
            iters=payload["iters"], reps=payload["reps"], seed=payload["seed"],
            record_every=payload["record_every"], device=DEVICE,
            store_reps=payload["store_reps"],
            progress=payload["progress"], progress_desc=f"fused {scen_tag}")

        results: list[tuple[str, dict]] = []
        for (gkey, n, nice_name), adj, sl in zip(blocks, adjs, ranges):
            topology_name = f"{gkey}_n{n}_g{gamma}_b{beta}_{learner}"
            out_dir = os.path.join(base_dir, topology_name)
            title = f"{nice_name} | n={n}, γ={gamma}, β={beta}, {learner}"
            try:
                summary = analyze_from_sim(
                    adj, sim.slice_nodes(sl),
                    out_dir, topology_name=topology_name, title=title,
                    seed=payload["seed"], n_final_steps=payload["n_final_steps"],
                    cluster_method=payload["cluster_method"],
                    save_data_artifacts=payload["save_data_artifacts"],
                    save_full_histories=payload["save_full_histories"],
                    graph_descriptor={"family": gkey, "n": n, "fused_scenario": scen_tag},
                    extra_summary={"graph_family": gkey, "n": n,
                                   "gamma": gamma, "beta": beta, "learner": learner})
                results.append((topology_name, {
                    "number_of_clusters": summary["number_of_clusters"],
                    "largest_cluster_fraction": summary["largest_cluster_fraction"],
                    "mean_cooperation": summary["mean_cooperation"],
                    "method_used": summary["method_used"],
                    "reps": payload["reps"], "fused": True,
                    "cluster_topology_correlation_eta": summary["cluster_topology_correlation_eta"],
                }))
            except Exception as exc:
                traceback.print_exc()
                results.append((topology_name, {"error": str(exc)}))
        return results
    except Exception as exc:  # whole-scenario failure
        traceback.print_exc()
        return [(f"FUSED_{scen_tag}", {"error": str(exc)})]


def _run_fused(args, scenarios, valid_graphs, base_dir, workers, inner_progress) -> None:
    """Drive the fused path: one block-diagonal simulation per scenario."""
    total_nodes = sum(n for _, n in valid_graphs)
    # In fused mode the batch lives on the full super-graph; size reps to it.
    if getattr(args, "auto_reps", False):
        reps = suggest_reps(total_nodes, device=DEVICE,
                            vram_fraction=args.vram_fraction / max(1, workers),
                            max_reps=args.max_reps)
    else:
        reps = args.reps

    payloads = [{
        "gamma": gamma, "beta": beta, "learner": learner,
        "graphs": list(valid_graphs), "base_dir": base_dir,
        "iters": args.iters, "reps": reps, "seed": args.seed,
        "record_every": args.record_every, "n_final_steps": args.n_final_steps,
        "cluster_method": args.cluster_method, "store_reps": args.store_reps,
        "save_data_artifacts": not args.no_data_artifacts,
        "save_full_histories": args.save_full_histories,
        "progress": inner_progress,
    } for (gamma, beta, learner) in scenarios]

    print("=" * 64)
    print(f"  FUSED mass run on device: {DEVICE}")
    print(f"  {len(valid_graphs)} graphs/scenario fused -> {total_nodes} nodes/sim")
    print(f"  |scenarios|={len(scenarios)}  iters={args.iters} reps={reps} "
          f"workers={workers}")
    print(f"  output: {base_dir}")
    print("=" * 64)

    index: dict[str, dict] = {}
    t0 = time.time()
    if workers == 1:
        for payload in tqdm(payloads, desc="scenarios", position=0):
            for name, entry in _run_fused_scenario(payload):
                index[name] = entry
                if "error" in entry:
                    print(f"  [error] {name}: {entry['error']}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = [ex.submit(_run_fused_scenario, p) for p in payloads]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="scenarios", position=0):
                for name, entry in fut.result():
                    index[name] = entry
                    if "error" in entry:
                        print(f"  [error] {name}: {entry['error']}")

    with open(os.path.join(base_dir, "index_summary.json"), "w") as f:
        json.dump(index, f, indent=2)
    elapsed = time.time() - t0
    print(f"\nDone (fused): {len(scenarios)} scenarios x {len(valid_graphs)} graphs "
          f"in {elapsed:.0f}s. Index: {os.path.join(base_dir, 'index_summary.json')}")


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
    parser.add_argument("--iters", type=int, default=500_000)
    parser.add_argument("--reps", type=int, default=4096,
                        help="batch size (overridden per-run when --auto-reps)")
    parser.add_argument("--auto-reps", action="store_true",
                        help="size reps to fill VRAM (A100); see --vram-fraction/--max-reps")
    parser.add_argument("--vram-fraction", type=float, default=0.85,
                        help="target fraction of free VRAM when --auto-reps")
    parser.add_argument("--max-reps", type=int, default=131_072,
                        help="upper cap for --auto-reps")
    parser.add_argument("--record-every", type=int, default=5_000)
    parser.add_argument("--n-final-steps", type=int, default=10_000)
    parser.add_argument("--cluster-method", default="auto",
                        choices=["auto", "dbscan", "hdbscan", "kmeans"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (share the GPU); 1 = sequential")
    parser.add_argument("--fuse-graphs", action="store_true",
                        help="run all graphs of a scenario in ONE block-diagonal "
                             "simulation (fills the GPU, far fewer Python loops). "
                             "Parallel unit becomes the scenario, not the graph.")
    parser.add_argument("--store-reps", default="reduced",
                        choices=["reduced", "full"],
                        help="keep only (T_out,N) mean/std, or full per-replicate histories")
    parser.add_argument("--progress", dest="progress", action="store_true", default=None,
                        help="force inner per-step tqdm bar")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="disable inner per-step tqdm bar")
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
        args.auto_reps = False
        args.record_every = 1_000
        args.n_final_steps = 2_000

    workers = max(1, args.workers)
    # Inner per-step bar is only legible with a single worker; default to that.
    inner_progress = (workers == 1) if args.progress is None else bool(args.progress)
    if args.progress and workers > 1:
        inner_progress = False  # avoid interleaved bars across processes

    base_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", "convergence_topology")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    scenarios = build_scenarios(args)

    # Valid (graph, n) pairs shared across scenarios.
    valid_graphs: list[tuple[str, int]] = []
    for gkey in args.graphs:
        _, _, valid = GRAPH_REGISTRY[gkey]
        for n in args.sizes:
            if valid(n):
                valid_graphs.append((gkey, n))
            else:
                print(f"  [skip] {gkey} n={n} (size constraint)")

    if args.fuse_graphs:
        _run_fused(args, scenarios, valid_graphs, base_dir, workers, inner_progress)
        return

    # Build one picklable payload per (graph, n, scenario).
    payloads: list[dict] = []
    for gkey, n in valid_graphs:
        reps = _resolve_reps(n, args, n_workers=workers)
        for gamma, beta, learner in scenarios:
            payloads.append({
                "gkey": gkey, "n": n, "gamma": gamma, "beta": beta,
                "learner": learner, "base_dir": base_dir, "iters": args.iters,
                "reps": reps, "seed": args.seed, "record_every": args.record_every,
                "n_final_steps": args.n_final_steps,
                "cluster_method": args.cluster_method,
                "store_reps": args.store_reps,
                "save_data_artifacts": not args.no_data_artifacts,
                "save_full_histories": args.save_full_histories,
                "progress": inner_progress,
            })

    print("=" * 64)
    print(f"  Mass convergence-topology run on device: {DEVICE}")
    print(f"  graphs={args.graphs} sizes={args.sizes}")
    print(f"  |scenarios|={len(scenarios)} (gamma x beta x learner)")
    print(f"  iters={args.iters} reps={'auto' if args.auto_reps else args.reps} "
          f"workers={workers} -> {len(payloads)} runs")
    print(f"  output: {base_dir}")
    print("=" * 64)

    index: dict[str, dict] = {}
    t0 = time.time()
    if workers == 1:
        for payload in tqdm(payloads, desc="topologies", position=0):
            name, entry = _run_one_config(payload)
            index[name] = entry
            if "error" in entry:
                print(f"  [error] {name}: {entry['error']}")
    else:
        ctx = mp.get_context("spawn")  # required for CUDA + multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = [ex.submit(_run_one_config, p) for p in payloads]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="topologies", position=0):
                name, entry = fut.result()
                index[name] = entry
                if "error" in entry:
                    print(f"  [error] {name}: {entry['error']}")

    with open(os.path.join(base_dir, "index_summary.json"), "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {len(payloads)} runs in {elapsed:.0f}s. Index: "
          f"{os.path.join(base_dir, 'index_summary.json')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"Fatal error: {e}")
        traceback.print_exc()

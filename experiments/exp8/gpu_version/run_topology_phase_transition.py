"""
Task 4 — topological phase-transition study.

Sweeps the topological ``temperature`` t in {0.00, 0.05, ..., 1.00} of the
continuous family produced by
:func:`analysis.interpolation.generate_interpolated_regular_graph` (which goes
from a k-regular graph at ``t=0`` to a (k+1)-regular graph at ``t=1``).

For each temperature it averages over several stochastic realizations and records:

* number of clusters
* cluster sizes (of the representative realization)
* mean Q-values (Q(C), Q(D))
* cooperation fraction
* degree distribution

Outputs (under ``results/phase_transition/<run_tag>/``):

    temp_0.00.png ... temp_1.00.png   graph coloured by convergence cluster
    phase_number_of_clusters.png
    phase_largest_cluster_fraction.png
    phase_mean_cooperation.png
    phase_summary.json                full per-temperature record
    phase_summary.csv                 tidy table for downstream analysis

Examples
--------
Quick smoke run::

    python -m experiments.exp8.gpu_version.run_topology_phase_transition --smoke

Default (k=2 -> 3, n=20)::

    python -m experiments.exp8.gpu_version.run_topology_phase_transition --n 20 --k 2

k=3 -> 4 with more realizations::

    python -m experiments.exp8.gpu_version.run_topology_phase_transition \\
        --n 20 --k 3 --realizations 5 --iters 300000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import experiments.exp8.gpu_version.utils.gpu_utils as gpu_utils  # noqa: E402

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
gpu_utils.gpu_config.device = DEVICE

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.exp8.gpu_version.analysis.interpolation import (  # noqa: E402
    generate_interpolated_regular_graph,
)
from experiments.exp8.gpu_version.analysis.pipeline import analyze_topology  # noqa: E402
from experiments.exp8.gpu_version.visualization.cluster_plotting import (  # noqa: E402
    plot_convergence_clusters,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def temperature_grid(step: float = 0.05) -> list[float]:
    """``[0.0, step, ..., 1.0]`` rounded to avoid float drift."""
    n_steps = int(round(1.0 / step))
    return [round(i * step, 4) for i in range(n_steps + 1)]


def _phase_plot(xs, ys, ylabel, title, save_path, *, color="#2980b9"):
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=6)
    plt.xlabel("temperature")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="number of vertices (even)")
    parser.add_argument("--k", type=int, default=2, help="base regularity (target k+1)")
    parser.add_argument("--step", type=float, default=0.05, help="temperature step")
    parser.add_argument("--realizations", type=int, default=3,
                        help="stochastic realizations to average per temperature")
    parser.add_argument("--mode", default="stochastic",
                        choices=["stochastic", "deterministic"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--learner", default="q_learning",
                        choices=["q_learning", "sarsa"])
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--reps", type=int, default=256)
    parser.add_argument("--record-every", type=int, default=5_000)
    parser.add_argument("--n-final-steps", type=int, default=10_000)
    parser.add_argument("--cluster-method", default="auto",
                        choices=["auto", "dbscan", "hdbscan", "kmeans"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout", default="circular",
                        choices=["circular", "spring", "kamada_kawai"])
    parser.add_argument("--save-full-histories", action="store_true",
                        help="store full (T_out, reps, N) histories in per-temp artifacts")
    parser.add_argument("--no-data-artifacts", action="store_true",
                        help="skip per-temperature artifacts.npz / run_params.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n = 10
        args.iters = 3_000
        args.reps = 16
        args.record_every = 1_000
        args.n_final_steps = 2_000
        args.realizations = 2
        args.step = 0.25

    run_tag = f"{args.mode}_n{args.n}_k{args.k}_g{args.gamma}_b{args.beta}_{args.learner}"
    base_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", "phase_transition", run_tag)
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    temps = temperature_grid(args.step)

    print("=" * 64)
    print(f"  Phase transition: {args.k}-regular -> {args.k + 1}-regular")
    print(f"  n={args.n} mode={args.mode} realizations={args.realizations}")
    print(f"  temperatures={temps}")
    print(f"  device={DEVICE}  output={base_dir}")
    print("=" * 64)

    records: list[dict] = []
    t0 = time.time()

    for temp in tqdm(temps, desc="temperature"):
        per_real_clusters: list[int] = []
        per_real_coop: list[float] = []
        per_real_qc: list[float] = []
        per_real_qd: list[float] = []
        per_real_largest: list[float] = []

        rep_summary = None
        rep_labels = None
        rep_adj = None
        rep_degrees = None

        for r in range(args.realizations):
            # Distinct, reproducible seed per (temperature, realization).
            real_seed = args.seed + r + int(round(temp * 1000))
            adj = generate_interpolated_regular_graph(
                args.n, args.k, temp, seed=real_seed,
                mode=args.mode, device=DEVICE)

            is_rep = (r == 0)
            # The representative realization (r=0) gets a full, reusable artifact
            # bundle in runs/t<temp>/ so its figures can be regenerated later.
            rep_out_dir = os.path.join(base_dir, "runs", f"t{temp:.2f}")
            result = analyze_topology(
                adj, rep_out_dir if is_rep else base_dir,
                topology_name=f"{run_tag}_t{temp:.2f}_r{r}",
                title=f"{run_tag} | t={temp:.2f}",
                gamma=args.gamma, beta=args.beta, learner_type=args.learner,
                iters=args.iters, reps=args.reps, seed=real_seed,
                record_every=args.record_every, n_final_steps=args.n_final_steps,
                cluster_method=args.cluster_method, device=DEVICE,
                save_artifacts=is_rep,
                save_data_artifacts=not args.no_data_artifacts,
                save_full_histories=args.save_full_histories,
                return_details=is_rep, layout=args.layout,
                graph_descriptor={"family": "interpolated", "n": args.n, "k": args.k,
                                  "temperature": temp, "mode": args.mode,
                                  "realization": r, "realization_seed": real_seed},
            )
            summary = result[0] if is_rep else result

            per_real_clusters.append(summary["number_of_clusters"])
            per_real_coop.append(summary["mean_cooperation"])
            per_real_qc.append(summary["mean_Q_C"])
            per_real_qd.append(summary["mean_Q_D"])
            per_real_largest.append(summary["largest_cluster_fraction"])

            if is_rep:
                rep_summary = summary
                rep_labels = result[1]["labels"]
                rep_adj = adj
                rep_degrees = result[1]["sim"].degrees

        # Representative-realization graph image: temp_<value>.png
        img_path = os.path.join(base_dir, f"temp_{temp:.2f}.png")
        plot_convergence_clusters(
            rep_adj, rep_labels, rep_degrees, img_path,
            title=f"{run_tag} | t={temp:.2f} | "
                  f"{rep_summary['number_of_clusters']} clusters",
            layout=args.layout)

        records.append({
            "temperature": temp,
            "number_of_clusters_mean": float(np.mean(per_real_clusters)),
            "number_of_clusters_rep": int(rep_summary["number_of_clusters"]),
            "largest_cluster_fraction_mean": float(np.mean(per_real_largest)),
            "mean_cooperation_mean": float(np.mean(per_real_coop)),
            "mean_cooperation_std": float(np.std(per_real_coop)),
            "mean_Q_C": float(np.mean(per_real_qc)),
            "mean_Q_D": float(np.mean(per_real_qd)),
            "cluster_sizes_rep": rep_summary["cluster_sizes"],
            "degree_distribution_rep": rep_summary["degree_distribution"],
            "cluster_topology_correlation_eta_rep":
                rep_summary["cluster_topology_correlation_eta"],
            "n_realizations": args.realizations,
        })

    # ── phase plots ──
    xs = [rec["temperature"] for rec in records]
    _phase_plot(xs, [rec["number_of_clusters_mean"] for rec in records],
                "number of clusters", f"{run_tag}: temperature vs #clusters",
                os.path.join(base_dir, "phase_number_of_clusters.png"))
    _phase_plot(xs, [rec["largest_cluster_fraction_mean"] for rec in records],
                "largest cluster fraction",
                f"{run_tag}: temperature vs largest-cluster fraction",
                os.path.join(base_dir, "phase_largest_cluster_fraction.png"),
                color="#8e44ad")
    _phase_plot(xs, [rec["mean_cooperation_mean"] for rec in records],
                "mean cooperation",
                f"{run_tag}: temperature vs mean cooperation",
                os.path.join(base_dir, "phase_mean_cooperation.png"),
                color="#27ae60")

    # ── persist summary ──
    with open(os.path.join(base_dir, "phase_summary.json"), "w") as f:
        json.dump({"run_tag": run_tag, "args": vars(args), "records": records},
                  f, indent=2)

    try:
        import pandas as pd
        flat = [{
            "temperature": r["temperature"],
            "number_of_clusters_mean": r["number_of_clusters_mean"],
            "number_of_clusters_rep": r["number_of_clusters_rep"],
            "largest_cluster_fraction_mean": r["largest_cluster_fraction_mean"],
            "mean_cooperation_mean": r["mean_cooperation_mean"],
            "mean_cooperation_std": r["mean_cooperation_std"],
            "mean_Q_C": r["mean_Q_C"],
            "mean_Q_D": r["mean_Q_D"],
        } for r in records]
        pd.DataFrame(flat).to_csv(
            os.path.join(base_dir, "phase_summary.csv"), index=False)
    except Exception as exc:
        print(f"  [warn] could not write phase_summary.csv: {exc}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(temps)} temperatures in {elapsed:.0f}s. Output: {base_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"Fatal error: {e}")
        traceback.print_exc()

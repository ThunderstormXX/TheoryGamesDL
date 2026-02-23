import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from game_launcher import PairGame
from graph_structure import SmallWorldGraph, StarGraph
from learner import QLearner
from reward_model import PPReward


def build_graph(name: str, n_nodes: int):
    if name == "small_world":
        return SmallWorldGraph(n_nodes, k=4, p=0.1)
    if name == "star":
        return StarGraph(n_nodes)
    raise ValueError(f"Unknown graph name: {name}")


def theoretical_mean_p(graph, c: float, temp: float):
    degrees = np.array(graph.get_degree(), dtype=float)
    return float(np.mean(1.0 / (1.0 + np.exp(c * degrees / temp))))


def run_one_trace(
    graph,
    b: float,
    c: float,
    gamma: float,
    temp: float,
    learning_rate: float,
    n_episodes: int,
    seed: int,
):
    np.random.seed(seed)
    n_nodes = len(graph.get_degree())
    learners = [
        QLearner(
            action_space_size=2,
            learning_rate=learning_rate,
            discount_factor=gamma,
            strategy="boltzmann",
            temperature=temp,
        )
        for _ in range(n_nodes)
    ]
    game = PairGame(graph, learners, PPReward(b=b, c=c))

    trace = np.zeros(n_episodes, dtype=float)
    for t in range(n_episodes):
        game.round()
        trace[t] = np.mean(game.strategies)
    return trace


def main():
    # Core setup
    n_nodes = 100
    b_values = [0.1, 0.5, 1.0, 5.0]
    conditions = [
        ("small_world", 1.0),
        ("small_world", 0.1),
        ("star", 1.0),
        ("star", 0.1),
    ]

    gamma = 0.0
    temp = 1.0
    learning_rate = 0.2
    n_episodes = 3000
    n_exps = 4
    tail = 500
    base_seed = 20260223

    out_dir = os.path.join("results", "gamma0_theory_diversity")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (graph_name, c) in enumerate(conditions):
        condition_name = f"{graph_name}_c{c}"
        print(f"Running condition: {condition_name}", flush=True)

        # Keep the same graph structure for all b inside a condition.
        np.random.seed(base_seed + idx * 10000)
        graph = build_graph(graph_name, n_nodes)
        theory = theoretical_mean_p(graph, c=c, temp=temp)

        ax = axes[idx]
        for b_idx, b in enumerate(b_values):
            traces = np.zeros((n_exps, n_episodes), dtype=float)
            for exp_id in range(n_exps):
                seed = base_seed + idx * 10000 + b_idx * 100 + exp_id
                traces[exp_id] = run_one_trace(
                    graph=graph,
                    b=b,
                    c=c,
                    gamma=gamma,
                    temp=temp,
                    learning_rate=learning_rate,
                    n_episodes=n_episodes,
                    seed=seed,
                )

            mean_trace = traces.mean(axis=0)
            std_trace = traces.std(axis=0)
            tail_per_exp = traces[:, -tail:].mean(axis=1)

            summary_rows.append(
                {
                    "condition": condition_name,
                    "graph": graph_name,
                    "c": c,
                    "b": b,
                    "gamma": gamma,
                    "temperature": temp,
                    "n_nodes": n_nodes,
                    "episodes": n_episodes,
                    "n_experiments": n_exps,
                    "theory_mean_p": theory,
                    "tail_mean": float(tail_per_exp.mean()),
                    "tail_std": float(tail_per_exp.std()),
                    "final_mean": float(traces[:, -1].mean()),
                }
            )

            ax.plot(mean_trace, label=f"b={b}")
            ax.fill_between(
                np.arange(n_episodes),
                mean_trace - std_trace,
                mean_trace + std_trace,
                alpha=0.12,
            )

        ax.axhline(
            y=theory,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"theory={theory:.3f}",
        )
        ax.set_title(f"{graph_name}, c={c}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cooperation rate")

    fig.suptitle(
        "PairGame, gamma=0: convergence for different b on fixed graph+c conditions",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig_path = os.path.join(out_dir, "convergence_vs_theory_all_conditions.png")
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:", fig_path)
    print("Saved:", csv_path)
    print("\nTail summary:")
    for row in summary_rows:
        print(
            f"{row['condition']}, b={row['b']}: tail={row['tail_mean']:.4f} "
            f"+- {row['tail_std']:.4f}, theory={row['theory_mean_p']:.4f}"
        )


if __name__ == "__main__":
    main()

"""
Comparison experiment: Q-learning vs LLM (in-context learning via OpenRouter)
on network Prisoner's Dilemma.

Experiments:
  EXP 1: Q-learning baseline
  EXP 2: LLM with "history_only" (II-style: sees own history + neighbor coop count)
  EXP 3: LLM with "history_and_global" (II + global cooperation fraction)
  EXP 4: LLM with "neighbors_detail" (ID-style: sees per-neighbor actions)

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python run_llm_openrouter_exp.py [--modes history_only history_and_global] [--episodes 100]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time
import json
import argparse
from datetime import datetime

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _this_dir)
sys.path.insert(0, _parent_dir)

from graph_structure import SmallWorldGraph, StarGraph
from learner import QLearner
from llm_agent_openrouter import LLMAgentOpenRouter
from reward_model import PPReward, PFReward, FFReward, FPReward
from game_launcher import PairGame
from llm_game_launcher import LLMPairGame


# ═══════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ═══════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "n_nodes": 8,
    "b": 3.0,
    "c": 1.0,
    "reward_type": "pp",
    "graph_type": "small_world",
    "graph_k": 4,
    "graph_p": 0.1,

    "ep_q": 500,
    "ep_llm": 100,

    "q_lr": 0.2,
    "q_gamma": 0.0,
    "q_temp": 1.0,
    "q_n_runs": 10,          # average Q-learning over multiple runs

    "llm_model": "mistralai/mistral-7b-instruct-v0.1",
    "llm_temperature": 0.2,
    "llm_max_history": 30,

    "seed": 42,
}

REWARD_CLASSES = {
    "pp": PPReward,
    "pf": PFReward,
    "ff": FFReward,
    "fp": FPReward,
}


def parse_args():
    p = argparse.ArgumentParser(description="Q-learning vs LLM on PD network")
    p.add_argument("--n_nodes", type=int, default=DEFAULT_CONFIG["n_nodes"])
    p.add_argument("--b", type=float, default=DEFAULT_CONFIG["b"])
    p.add_argument("--c", type=float, default=DEFAULT_CONFIG["c"])
    p.add_argument("--reward_type", choices=REWARD_CLASSES.keys(),
                   default=DEFAULT_CONFIG["reward_type"])
    p.add_argument("--graph_type", choices=["small_world", "star"],
                   default=DEFAULT_CONFIG["graph_type"])
    p.add_argument("--ep_q", type=int, default=DEFAULT_CONFIG["ep_q"])
    p.add_argument("--ep_llm", type=int, default=DEFAULT_CONFIG["ep_llm"])
    p.add_argument("--q_n_runs", type=int, default=DEFAULT_CONFIG["q_n_runs"])
    p.add_argument("--q_lr", type=float, default=DEFAULT_CONFIG["q_lr"],
                   help="Q-learning learning rate alpha")
    p.add_argument("--q_gamma", type=float, default=DEFAULT_CONFIG["q_gamma"],
                   help="Q-learning discount factor gamma")
    p.add_argument("--q_temp", type=float, default=DEFAULT_CONFIG["q_temp"],
                   help="Q-learning Boltzmann temperature")
    p.add_argument("--llm_model", type=str, default=DEFAULT_CONFIG["llm_model"])
    p.add_argument("--llm_temperature", type=float,
                   default=DEFAULT_CONFIG["llm_temperature"])
    p.add_argument("--modes", nargs="+",
                   default=["history_only", "history_and_global", "neighbors_detail", "blind"],
                   help="Which LLM prompt modes to run")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--output_dir", type=str, default="results/llm_exp")
    p.add_argument("--api_key", type=str, default=None,
                   help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    p.add_argument("--verbose_llm", action="store_true",
                   help="Print every LLM prompt and response to stdout")
    p.add_argument("--api_delay", type=float, default=1.0,
                   help="Seconds to wait between API calls (default 1.0, for rate limiting)")
    return p.parse_args()


def make_graph(graph_type, n_nodes, k=4, p=0.1):
    if graph_type == "small_world":
        return SmallWorldGraph(n_nodes, k=k, p=p)
    elif graph_type == "star":
        return StarGraph(n_nodes)
    raise ValueError(f"Unknown graph type: {graph_type}")


def play(game, n_episodes, label="", verbose_every=20):
    """Run game for n_episodes, return cooperation rates and reward traces."""
    rates = []
    mean_rewards = []
    for ep in range(n_episodes):
        game.round()
        rho = float(np.mean(game.strategies))
        rates.append(rho)
        r = game.history[-1]["rewards"]
        mean_rewards.append(float(np.mean(r)))
        if label and verbose_every and (ep + 1) % verbose_every == 0:
            print(f"  [{label}] ep {ep+1:>4}/{n_episodes}  "
                  f"ρ={rho:.3f}  avg_r={mean_rewards[-1]:.2f}")
    return np.array(rates), np.array(mean_rewards)


def run_q_learning(graph, reward_model, config, init_strat, n_runs=10):
    """Run Q-learning multiple times and return mean/std traces."""
    all_rates = []
    all_rewards = []
    for run in range(n_runs):
        learners = [
            QLearner(
                action_space_size=2,
                learning_rate=config["q_lr"],
                discount_factor=config["q_gamma"],
                strategy="boltzmann",
                temperature=config["q_temp"],
            )
            for _ in range(config["n_nodes"])
        ]
        game = PairGame(graph, learners, reward_model)
        game.strategies = init_strat.copy()
        rates, rewards = play(game, config["ep_q"],
                              label=f"Q-run{run+1}" if run == 0 else "",
                              verbose_every=100 if run == 0 else 0)
        all_rates.append(rates)
        all_rewards.append(rewards)
    return np.array(all_rates), np.array(all_rewards)


def run_llm_experiment(graph, reward_model, config, init_strat, mode, api_key,
                      verbose=False, api_delay=1.0, log_path=None):
    """Run a single LLM experiment with given prompt_mode."""
    degrees = graph.get_degree()
    neighbors_map = graph.get_neibhours()

    agents = []
    for i in range(config["n_nodes"]):
        agent = LLMAgentOpenRouter(
            agent_id=i,
            degree=degrees[i],
            model=config["llm_model"],
            temperature=config["llm_temperature"],
            max_history=config["llm_max_history"],
            api_key=api_key,
            prompt_mode=mode,
            neighbor_ids=neighbors_map.get(i, []),
            verbose=verbose,
            api_delay=api_delay,
        )
        if log_path is not None:
            agent.set_log_file(log_path)
        agents.append(agent)

    game = LLMPairGame(graph, agents, reward_model)
    game.strategies = init_strat.copy()

    t0 = time.time()
    rates, rewards = play(game, config["ep_llm"], label=f"LLM-{mode}")
    elapsed = time.time() - t0

    # Collect agent stats
    agent_stats = [a.get_stats() for a in agents]
    total_calls = sum(s["api_calls"] for s in agent_stats)
    total_errors = sum(s["api_errors"] for s in agent_stats)

    print(f"  [{mode}] done in {elapsed:.0f}s  "
          f"({total_calls} API calls, {total_errors} errors)")

    return rates, rewards, elapsed, agent_stats


def _build_info_text(config, results):
    """Build a compact multi-line string of all hyperparameters for plot footer."""
    gt = config["graph_type"]
    if gt == "small_world":
        graph_str = f"Small-World (k={config['graph_k']}, p={config['graph_p']})"
    elif gt == "star":
        graph_str = "Star"
    else:
        graph_str = gt

    line1 = (f"Graph: {graph_str},  N={config['n_nodes']}  |  "
             f"Reward: {config['reward_type']},  b={config['b']},  c={config['c']}")

    line2 = (f"Q-learning: α={config['q_lr']},  γ={config['q_gamma']},  "
             f"T={config['q_temp']},  episodes={config['ep_q']},  "
             f"runs={config['q_n_runs']}")

    llm_modes = sorted(k.replace("llm_", "") for k in results if k.startswith("llm_"))
    if llm_modes:
        line3 = (f"LLM: {config['llm_model']},  T_llm={config.get('llm_temperature', '?')},  "
                 f"history={config.get('llm_max_history', '?')},  "
                 f"episodes={config['ep_llm']},  "
                 f"modes: [{', '.join(llm_modes)}]")
    else:
        line3 = "LLM: not run"

    return f"{line1}\n{line2}\n{line3}"


def _add_info_box(fig, config, results):
    """Add a text box with hyperparameters at the bottom of a figure."""
    txt = _build_info_text(config, results)
    fig.text(
        0.5, -0.01, txt,
        ha="center", va="top", fontsize=8,
        fontstyle="italic", color="gray",
        family="monospace",
    )


def _make_run_tag(config, results):
    """Build a short unique tag from parameters + timestamp for filenames."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes = sorted(k.replace("llm_", "") for k in results if k.startswith("llm_"))
    modes_str = "+".join(modes) if modes else "noLLM"
    return (
        f"{ts}"
        f"_N{config['n_nodes']}"
        f"_{config['graph_type']}"
        f"_{config['reward_type']}"
        f"_b{config['b']}_c{config['c']}"
        f"_g{config['q_gamma']}_T{config['q_temp']}"
        f"_{modes_str}"
    )


def plot_results(results, config, output_dir):
    """Create comparison plots with full hyperparameter annotations."""
    coop_dir = os.path.join(output_dir, "cooperation")
    rewards_dir = os.path.join(output_dir, "rewards")
    os.makedirs(coop_dir, exist_ok=True)
    os.makedirs(rewards_dir, exist_ok=True)
    tag = _make_run_tag(config, results)

    colors = {"history_only": "coral", "history_and_global": "green",
              "neighbors_detail": "purple", "blind": "orange"}

    # ── Suptitle ──
    gt = config["graph_type"]
    if gt == "small_world":
        graph_label = f"Small-World (k={config['graph_k']}, p={config['graph_p']})"
    elif gt == "star":
        graph_label = "Star"
    else:
        graph_label = gt
    suptitle = (f"N={config['n_nodes']},  {graph_label},  "
                f"reward={config['reward_type']},  b={config['b']},  c={config['c']}")

    # ══════════════════════════════════════════════════════════
    #  Plot 1: Cooperation dynamics + bar chart
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(f"Q-learning vs LLM  —  {suptitle}",
                 fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0]
    if "q_learning" in results:
        q = results["q_learning"]
        q_mean = np.mean(q["rates"], axis=0)
        q_std = np.std(q["rates"], axis=0)
        ax.plot(q_mean,
                label=(f"Q-learning (α={config['q_lr']}, γ={config['q_gamma']}, "
                       f"T={config['q_temp']}, {q['rates'].shape[0]} runs)"),
                color="steelblue", linewidth=2)
        ax.fill_between(range(len(q_mean)), q_mean - q_std, q_mean + q_std,
                        alpha=0.2, color="steelblue")

    for mode, data in results.items():
        if mode.startswith("llm_"):
            mname = mode.replace("llm_", "")
            c = colors.get(mname, "gray")
            ax.plot(data["rates"], label=f"LLM ({mname})", color=c,
                    linewidth=2, alpha=0.9)

    if "rho_theory" in results:
        ax.axhline(results["rho_theory"], color="red", ls="--", lw=1.5,
                   label=f"Theory ρ̂={results['rho_theory']:.3f}")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cooperation Rate", fontsize=12)
    ax.set_title("Cooperation Dynamics", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Bar chart
    ax = axes[1]
    names, vals, errs, bar_colors = [], [], [], []

    if "q_learning" in results:
        q = results["q_learning"]
        q_finals = np.mean(q["rates"][:, -50:], axis=1)
        names.append("Q-learning")
        vals.append(np.mean(q_finals))
        errs.append(np.std(q_finals))
        bar_colors.append("steelblue")

    for mode in ["history_only", "history_and_global", "neighbors_detail", "blind"]:
        key = f"llm_{mode}"
        if key in results:
            d = results[key]
            final_val = float(np.mean(d["rates"][-50:]))
            names.append(f"LLM\n{mode}")
            vals.append(final_val)
            errs.append(0)
            bar_colors.append(colors.get(mode, "gray"))

    if "rho_theory" in results:
        names.append("Theory")
        vals.append(results["rho_theory"])
        errs.append(0)
        bar_colors.append("red")

    bars = ax.bar(names, vals, yerr=errs, color=bar_colors, alpha=0.8, capsize=5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Final Cooperation Rate", fontsize=12)
    ax.set_title("Steady-state Comparison", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    _add_info_box(fig, config, results)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fname = os.path.join(coop_dir, f"cooperation_{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot → {fname}")

    # ══════════════════════════════════════════════════════════
    #  Plot 2: Reward dynamics
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Average Reward per Round  —  {suptitle}",
                 fontsize=14, fontweight="bold", y=0.98)

    if "q_learning" in results:
        q = results["q_learning"]
        q_mean_r = np.mean(q["rewards"], axis=0)
        ax.plot(q_mean_r,
                label=f"Q-learning (α={config['q_lr']}, γ={config['q_gamma']}, T={config['q_temp']})",
                color="steelblue", linewidth=2)

    for mode in ["history_only", "history_and_global", "neighbors_detail", "blind"]:
        key = f"llm_{mode}"
        if key in results:
            d = results[key]
            ax.plot(d["rewards"], label=f"LLM ({mode})",
                    color=colors.get(mode, "gray"), linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Reward Dynamics", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    _add_info_box(fig, config, results)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fname = os.path.join(rewards_dir, f"reward_{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {fname}")


def main():
    args = parse_args()
    config = {
        "n_nodes": args.n_nodes,
        "b": args.b,
        "c": args.c,
        "reward_type": args.reward_type,
        "graph_type": args.graph_type,
        "graph_k": DEFAULT_CONFIG["graph_k"],
        "graph_p": DEFAULT_CONFIG["graph_p"],
        "ep_q": args.ep_q,
        "ep_llm": args.ep_llm,
        "q_lr": args.q_lr,
        "q_gamma": args.q_gamma,
        "q_temp": args.q_temp,
        "q_n_runs": args.q_n_runs,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_max_history": DEFAULT_CONFIG["llm_max_history"],
        "seed": args.seed,
    }

    np.random.seed(config["seed"])
    os.makedirs(args.output_dir, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")

    print("=" * 60)
    print("  Q-learning vs LLM on Prisoner's Dilemma Network")
    print("=" * 60)
    print(f"  Nodes:       {config['n_nodes']}")
    print(f"  Graph:       {config['graph_type']}")
    print(f"  Reward:      {config['reward_type']} (b={config['b']}, c={config['c']})")
    print(f"  Q episodes:  {config['ep_q']}  (x{config['q_n_runs']} runs)")
    print(f"  LLM episodes:{config['ep_llm']}")
    print(f"  LLM model:   {config['llm_model']}")
    print(f"  LLM modes:   {args.modes}")
    print("=" * 60)

    # ── Setup ──
    graph = make_graph(config["graph_type"], config["n_nodes"],
                       config["graph_k"], config["graph_p"])
    reward_cls = REWARD_CLASSES[config["reward_type"]]
    reward_model = reward_cls(b=config["b"], c=config["c"])
    degrees = graph.get_degree()
    init_strat = np.random.randint(0, 2, size=config["n_nodes"])

    # Theoretical cooperation rate (Boltzmann, gamma=0)
    rho_theory = float(np.mean(
        [1.0 / (1.0 + np.exp(config["c"] * k / config["q_temp"]))
         for k in degrees]
    ))
    print(f"\nTheoretical ρ̂ = {rho_theory:.4f}")

    results = {"rho_theory": rho_theory}

    # ── Q-learning ──
    print(f"\n{'─'*40}")
    print("▶ Q-learning baseline")
    q_rates, q_rewards = run_q_learning(
        graph, reward_model, config, init_strat, n_runs=config["q_n_runs"]
    )
    results["q_learning"] = {
        "rates": q_rates,
        "rewards": q_rewards,
    }
    q_final = float(np.mean(q_rates[:, -50:]))
    print(f"  Q-learning final ρ = {q_final:.4f}")

    # ── LLM experiments ──
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for mode in args.modes:
        print(f"\n{'─'*40}")
        print(f"▶ LLM mode: {mode}")

        if not api_key:
            print("  ⚠ No OPENROUTER_API_KEY set, skipping LLM experiments.")
            break

        log_path = os.path.join(logs_dir, f"log_{run_ts}_{mode}.txt")
        rates, rewards, elapsed, agent_stats = run_llm_experiment(
            graph, reward_model, config, init_strat, mode, api_key,
            verbose=args.verbose_llm,
            api_delay=args.api_delay,
            log_path=log_path,
        )
        print(f"  Log → {log_path}")
        results[f"llm_{mode}"] = {
            "rates": rates,
            "rewards": rewards,
            "elapsed": elapsed,
            "agent_stats": agent_stats,
        }

    # ── Save ──
    tag = _make_run_tag(config, results)

    # Save raw numeric data
    save_data = {
        "config": config,
        "rho_theory": rho_theory,
        "init_strat": init_strat,
        "degrees": degrees,
    }
    if "q_learning" in results:
        save_data["q_rates"] = results["q_learning"]["rates"]
        save_data["q_rewards"] = results["q_learning"]["rewards"]
    for mode in args.modes:
        key = f"llm_{mode}"
        if key in results:
            save_data[f"{key}_rates"] = results[key]["rates"]
            save_data[f"{key}_rewards"] = results[key]["rewards"]

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, f"data_{tag}.npz"), **save_data)

    # Save config + stats as JSON
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "rho_theory": rho_theory,
        "modes_run": args.modes,
    }
    if "q_learning" in results:
        json_report["q_final_coop"] = q_final
    for mode in args.modes:
        key = f"llm_{mode}"
        if key in results:
            d = results[key]
            json_report[f"{key}_final_coop"] = float(np.mean(d["rates"][-50:]))
            json_report[f"{key}_elapsed_s"] = d["elapsed"]
            json_report[f"{key}_total_api_calls"] = sum(
                s["api_calls"] for s in d["agent_stats"]
            )

    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, f"report_{tag}.json"), "w") as f:
        json.dump(json_report, f, indent=2)

    # ── Plot ──
    plot_results(results, config, args.output_dir)

    print(f"\n{'='*60}")
    print("  All results saved to:", args.output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
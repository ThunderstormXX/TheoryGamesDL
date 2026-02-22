# time_testing/test_exit_time.py

import sys
import os
import tempfile

# Исправляем проблему с runtime directory в WSL
_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
if not _runtime_dir or not os.path.isdir(_runtime_dir) or (os.stat(_runtime_dir).st_mode & 0o777) != 0o700:
    tmp_runtime = os.path.join(tempfile.gettempdir(), f"runtime-{os.getuid()}")
    os.makedirs(tmp_runtime, exist_ok=True)
    try:
        os.chmod(tmp_runtime, 0o700)
    except PermissionError:
        pass
    os.environ["XDG_RUNTIME_DIR"] = tmp_runtime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Optional
from tqdm import trange
import matplotlib
matplotlib.use('Agg')  # для WSL
import matplotlib.pyplot as plt


from bots import BoltzmannAgent
from environment import GameFactory


# =========================
# Trap detection parameters
# =========================

P_DEFECT_EPS = 0.05
P_COOP_DELTA = 0.1
WINDOW = 200
MAX_TIME = 5_000_000


# =========================
# Trap detectors
# =========================

def in_trap(p_hist: np.ndarray) -> bool:
    return np.all(p_hist.mean(axis=1) < P_DEFECT_EPS)


def out_of_trap(p_hist: np.ndarray) -> bool:
    return np.all(p_hist.mean(axis=1) > P_COOP_DELTA)


# =========================
# Single run
# =========================

def measure_exit_time(
    *,
    seed: int,
    n_players: int,
    alpha: float,
    beta: float,
    gamma: float,
    benefit: float,
    cost: float,
    reward_offset: float,
    init_q: Optional[List[float]],
) -> Optional[int]:

    game = GameFactory.create_generalized_prisoners_dilemma(
        n_players=n_players,
        benefit=benefit,
        cost=cost,
        reward_offset=reward_offset,
    )

    agents = [
        BoltzmannAgent(
            name=f"A{i}",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            init_q=init_q,
            seed=seed + i,
        )
        for i in range(n_players)
    ]

    p_buffer = np.zeros((n_players, WINDOW), dtype=float)

    entered_trap = False
    t_enter = None

    for t in range(MAX_TIME):
        actions = [agent.choose_action() for agent in agents]
        rewards = game.get_payoffs(tuple(actions))

        for i, agent in enumerate(agents):
            agent.learn(actions[i], rewards[i])
            p_buffer[i, t % WINDOW] = agent.current_p_cooperate()

        if t < WINDOW:
            continue

        window_view = np.roll(p_buffer, -((t + 1) % WINDOW), axis=1)

        if not entered_trap and in_trap(window_view):
            entered_trap = True
            t_enter = t
            continue

        if entered_trap and out_of_trap(window_view):
            return t - t_enter

    return None


# =========================
# Multiple runs
# =========================

def run_experiment(*, n_runs: int, **params) -> List[int]:
    taus = []

    for seed in trange(n_runs, desc="Exit time experiments"):
        tau = measure_exit_time(seed=seed, **params)
        if tau is not None:
            taus.append(tau)

    return taus


def estimate_mean_exit_time(n_runs: int, **params):
    taus = run_experiment(n_runs=n_runs, **params)

    if len(taus) == 0:
        return {
            "mean_tau": np.nan,
            "log_mean_tau": np.nan,
            "std_tau": np.nan,
            "n_valid": 0,
        }

    taus = np.array(taus)
    mean_tau = np.mean(taus)

    return {
        "mean_tau": mean_tau,
        "log_mean_tau": np.log(mean_tau),
        "std_tau": np.std(taus),
        "n_valid": len(taus),
    }


# =========================
# Main: theory checks
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exit time experiments")
    parser.add_argument("--beta", action="store_true", help="Run beta experiment")
    parser.add_argument("--depth", action="store_true", help="Run depth experiment")
    parser.add_argument("--theory", action="store_true", help="Run theory collapse experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--n_runs", type=int, default=6, help="Number of runs per parameter")
    
    args = parser.parse_args()
    
    # Если ничего не выбрано, запускаем все
    if not (args.beta or args.depth or args.theory or args.all):
        args.all = True
    
    if args.all:
        args.beta = args.depth = args.theory = True

    # =========================
    # Base parameters
    # =========================

    base_params = dict(
        n_players=2,
        alpha=0.001,
        benefit=6.0,
        cost=4.0,
        reward_offset=1.0,
    )

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ============================================================
    # 1. log <tau> vs beta (policy sharpness, not theory-critical)
    # ============================================================
    
    if args.beta:
        print("Running beta experiment...")
        betas = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        log_taus_beta = []

        for beta in betas:
            res = estimate_mean_exit_time(
                n_runs=args.n_runs,
                beta=beta,
                gamma=0.7,
                init_q=[-2.0, 2.0],
                **base_params,
            )
            log_taus_beta.append(res["log_mean_tau"])

        plt.figure()
        plt.plot(betas, log_taus_beta, marker="o")
        plt.xlabel(r"Inverse temperature $\beta$")
        plt.ylabel(r"$\log \langle \tau \rangle$")
        plt.title(r"Exit time vs inverse temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "log_tau_vs_beta.png"), dpi=150)
        plt.close()
        print(f"Beta experiment saved: {os.path.join(results_dir, 'log_tau_vs_beta.png')}")

    # ==========================================
    # 2. log <tau> vs trap depth d
    # ==========================================
    
    if args.depth:
        print("Running depth experiment...")
        depths = np.array([1.0, 1.5, 2.0])
        log_taus_depth = []

        for d in depths:
            res = estimate_mean_exit_time(
                n_runs=args.n_runs,
                beta=1.5,
                gamma=0.7,
                init_q=[-d, d],
                **base_params,
            )
            log_taus_depth.append(res["log_mean_tau"])

        plt.figure()
        plt.plot(depths, log_taus_depth, marker="o")
        plt.xlabel(r"Trap depth $d$")
        plt.ylabel(r"$\log \langle \tau \rangle$")
        plt.title(r"Exit time vs trap depth")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "log_tau_vs_depth.png"), dpi=150)
        plt.close()
        print(f"Depth experiment saved: {os.path.join(results_dir, 'log_tau_vs_depth.png')}")

    # ==========================================================
    # 3. DIRECT THEORY CHECK:
    #    log <tau> vs d^2 / (alpha * gamma^2)
    # ==========================================================
    
    if args.theory:
        print("Running theory collapse experiment...")
        gammas = np.linspace(0.05, 0.4, 3)
        depths = np.array([1.0, 1.2])

        X_vals = []
        Y_vals = []

        for gamma in gammas:
            for d in depths:
                res = estimate_mean_exit_time(
                    n_runs=args.n_runs,
                    beta=1.5,          # фиксируем политику
                    gamma=gamma,
                    init_q=[-d, d],
                    **base_params,
                )

                if res["n_valid"] == 0:
                    continue

                X = (d ** 2) / (base_params["alpha"] * gamma ** 2)
                Y = res["log_mean_tau"]

                X_vals.append(X)
                Y_vals.append(Y)

        X_vals = np.array(X_vals)
        Y_vals = np.array(Y_vals)

        # линейная аппроксимация
        coeffs = np.polyfit(X_vals, Y_vals, 1)
        X_line = np.linspace(X_vals.min(), X_vals.max(), 200)
        Y_line = coeffs[0] * X_line + coeffs[1]

        plt.figure()
        plt.scatter(X_vals, Y_vals, label="Simulation")
        plt.plot(X_line, Y_line, linestyle="--", label="Linear fit")
        plt.xlabel(r"$d^2 / (\alpha \gamma^2)$")
        plt.ylabel(r"$\log \langle \tau \rangle$")
        plt.title(r"Direct test of exit-time theory")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "theory_collapse.png"), dpi=150)
        plt.close()
        print(f"Theory experiment saved: {os.path.join(results_dir, 'theory_collapse.png')}")

    print(f"\nAll selected experiments completed. Results in: {results_dir}")


import argparse
import numpy as np
from typing import List

from .continuous_game import ContinuousBimatrixGame, PayoffParams
from .softmax_sarsa_agent import SoftmaxSARSAAgent
from .viz import plot_two_policies_heatmaps_over_time

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x):
        return x


def run(r: float, p: float, q: float, s: float, n: int, steps: int, alpha: float, gamma: float, beta: float, seed: int,
    init_mode: str = "uniform", init_action: int | None = None, init_epsilon: float = 1e-3):
    params = PayoffParams(r=r, p=p, q=q, s=s)
    if not params.is_prisoners_dilemma():
        print("⚠️ Параметры не удовлетворяют T>R>P>S (PD). Продолжаем, но проверьте r,p,q,s.")
    game = ContinuousBimatrixGame(params, n=n)

    agent_a = SoftmaxSARSAAgent(num_actions=game.num_actions(), alpha=alpha, gamma=gamma, beta=beta, seed=seed,
                                init_mode=init_mode, init_action=init_action, init_epsilon=init_epsilon)
    agent_b = SoftmaxSARSAAgent(num_actions=game.num_actions(), alpha=alpha, gamma=gamma, beta=beta, seed=seed + 1,
                                init_mode=init_mode, init_action=init_action, init_epsilon=init_epsilon)

    policies_a: List[np.ndarray] = []
    policies_b: List[np.ndarray] = []

    # стартовые действия
    a = agent_a.start_episode()
    b = agent_b.start_episode()

    for t in tqdm(range(steps)):
        # вознаграждения за текущие действия
        r_a = game.payoff_player0(a, b)
        r_b = game.payoff_player1(a, b)
        # следующий шаг: выбираем next_action для SARSA
        next_a = agent_a.choose_action()
        next_b = agent_b.choose_action()
        # SARSA-обновления
        agent_a.step(r_a, next_action=next_a)
        agent_b.step(r_b, next_action=next_b)
        # фиксируем распределения
        policies_a.append(agent_a.get_action_probs())
        policies_b.append(agent_b.get_action_probs())
        # продвигаем действия
        a, b = next_a, next_b

    plot_two_policies_heatmaps_over_time(policies_a, policies_b, title_a="Agent A", title_b="Agent B")


def main():
    parser = argparse.ArgumentParser(description="Exp4: continuous actions via discretization + Softmax-SARSA")
    parser.add_argument("--r", type=float, default=5.0, help="coef for a*b (R)")
    parser.add_argument("--p", type=float, default=0.0, help="coef for (1-a)*b (S)")
    parser.add_argument("--q", type=float, default=10.0, help="coef for a*(1-b) (T)")
    parser.add_argument("--s", type=float, default=1.0, help="coef for (1-a)*(1-b) (P)")
    parser.add_argument("--n", type=int, default=20, help="discretization steps (n => n+1 actions)")
    parser.add_argument("--steps", type=int, default=2000, help="time steps")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount")
    parser.add_argument("--beta", type=float, default=5.0, help="softmax beta")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--init-mode", type=str, default="uniform", choices=["uniform", "delta"],
                        help="initial policy mode: uniform or near-delta")
    parser.add_argument("--init-action", type=int, default=None, help="target action index for delta-like init")
    parser.add_argument("--init-epsilon", type=float, default=1e-3,
                        help="mass outside target for delta-like init (distributed over others)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    run(r=args.r, p=args.p, q=args.q, s=args.s, n=args.n, steps=args.steps,
        alpha=args.alpha, gamma=args.gamma, beta=args.beta, seed=args.seed,
        init_mode=args.init_mode, init_action=args.init_action, init_epsilon=args.init_epsilon)


if __name__ == "__main__":
    main()

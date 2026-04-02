"""Wrapper around theorygamesdl.utils.simulation.simulate for exp10."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from theorygamesdl.utils.simulation import simulate


@dataclass(frozen=True)
class SimConfig:
    pd: list[float]
    time: int
    gamma: float
    alpha: float
    beta: float
    seed: int | None = None
    mode: str = "donation"
    grid_type: str = "donation"
    B: float | None = None
    C: float | None = None


@contextmanager
def temporary_seed(seed: int | None):
    """Temporarily set random state for reproducible runs."""
    if seed is None:
        yield
        return

    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def payoff_gaps(pd: list[float]) -> dict[str, float]:
    """Return gaps from payoff vector [R, P, S, T]."""
    r, p, s, t = [float(x) for x in pd]
    return {
        "g1": p - s,
        "g2": r - p,
        "g3": t - r,
    }


def is_konstantinov_compatible(pd: list[float], tol: float = 1e-9) -> tuple[bool, float | None, float | None]:
    """Check whether pd has donation form [B-C, 0, -C, B]."""
    r, p, s, t = [float(x) for x in pd]
    if abs(p) > tol:
        return False, None, None

    c = -s
    b = t
    expected_r = b - c
    if abs(r - expected_r) > tol or not (b > c):
        return False, None, None
    return True, b, c


def run_simulation(config: SimConfig) -> dict[str, Any]:
    """Run a single simulation and return standardized output dict."""
    with temporary_seed(config.seed):
        pol1_y1, pol2_y1, q1c, q1d, q2c, q2d, h_act, h_rew = simulate(
            pd=config.pd,
            time=config.time,
            gamma=config.gamma,
            alpha=config.alpha,
            beta=config.beta,
            show_q=False,
            show_plots=False,
        )

    gaps = payoff_gaps(config.pd)
    konst, b_val, c_val = is_konstantinov_compatible(config.pd)

    return {
        "prob_c_player1": [float(x) for x in pol1_y1],
        "prob_c_player2": [float(x) for x in pol2_y1],
        "Q-val_player1_act_c": [float(x) for x in q1c],
        "Q-val_player1_act_d": [float(x) for x in q1d],
        "Q-val_player2_act_c": [float(x) for x in q2c],
        "Q-val_player2_act_d": [float(x) for x in q2d],
        "history_actions": h_act,
        "history_rewards": h_rew,
        "meta": {
            "pd": [float(x) for x in config.pd],
            "time": int(config.time),
            "gamma": float(config.gamma),
            "alpha": float(config.alpha),
            "beta": float(config.beta),
            "seed": None if config.seed is None else int(config.seed),
            "mode": config.mode,
            "grid_type": config.grid_type,
            "B": None if config.B is None else float(config.B),
            "C": None if config.C is None else float(config.C),
            "gaps": gaps,
            "is_konstantinov": konst,
            "konstantinov_B": b_val,
            "konstantinov_C": c_val,
        },
    }

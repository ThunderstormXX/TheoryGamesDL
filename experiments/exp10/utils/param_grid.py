"""Parameter grid builders for exp10 trap search."""

from __future__ import annotations

import itertools
from collections import OrderedDict


DEFAULT_GAMMA = [0.85, 0.9, 0.95]
DEFAULT_BETA = [1.0, 1.5, 2.0, 3.0]
DEFAULT_ALPHA = [0.005, 0.01, 0.02]
DEFAULT_TIME = [100000, 200000]
DEFAULT_C = [1, 2, 3, 4]
DEFAULT_B_DELTAS = [1, 2, 3]
DEFAULT_B_AFFINE = [
    (2, 1),  # 2*C + 1
    (2, 2),  # 2*C + 2
]

BASELINE_PD = [
    [3, 1, 0, 4],
    [2, 0, -1, 3],
    [3, 0, -2, 5],
    [4, 0, -3, 7],
    [6, 0, -4, 10],
]


def donation_pd(b: float, c: float) -> list[float]:
    """Build donation-game payoff [R, P, S, T] = [B-C, 0, -C, B]."""
    return [float(b - c), 0.0, float(-c), float(b)]


def _unique_preserve_order(values: list[float]) -> list[float]:
    return list(OrderedDict((float(x), None) for x in values).keys())


def donation_b_values(c: float) -> list[float]:
    """B candidates for a fixed C from task statement."""
    vals: list[float] = [c + d for d in DEFAULT_B_DELTAS]
    vals.extend([a * c + b for a, b in DEFAULT_B_AFFINE])
    vals = [float(v) for v in vals if v > c]
    return _unique_preserve_order(vals)


def donation_grid(
    gamma_values: list[float],
    beta_values: list[float],
    alpha_values: list[float],
    time_values: list[int],
    c_values: list[float],
) -> list[dict]:
    """Build grid for Konstantinov-compatible donation matrices."""
    configs: list[dict] = []
    for c in c_values:
        for b in donation_b_values(c):
            pd = donation_pd(b, c)
            for gamma, beta, alpha, time in itertools.product(
                gamma_values, beta_values, alpha_values, time_values
            ):
                configs.append(
                    {
                        "grid_type": "donation",
                        "pd": pd,
                        "B": float(b),
                        "C": float(c),
                        "gamma": float(gamma),
                        "beta": float(beta),
                        "alpha": float(alpha),
                        "time": int(time),
                    }
                )
    return configs


def baseline_grid(
    gamma_values: list[float],
    beta_values: list[float],
    alpha_values: list[float],
    time_values: list[int],
    baseline_pd: list[list[float]] | None = None,
) -> list[dict]:
    """Build handcrafted baseline grid over several PD matrices."""
    if baseline_pd is None:
        baseline_pd = BASELINE_PD

    configs: list[dict] = []
    for pd in baseline_pd:
        for gamma, beta, alpha, time in itertools.product(
            gamma_values, beta_values, alpha_values, time_values
        ):
            configs.append(
                {
                    "grid_type": "baseline",
                    "pd": [float(x) for x in pd],
                    "B": None,
                    "C": None,
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "alpha": float(alpha),
                    "time": int(time),
                }
            )
    return configs


def build_grid(
    mode: str,
    gamma_values: list[float],
    beta_values: list[float],
    alpha_values: list[float],
    time_values: list[int],
    c_values: list[float],
    baseline_pd: list[list[float]] | None = None,
) -> list[dict]:
    """Build final search grid depending on selected mode."""
    mode = mode.lower()
    if mode not in {"donation", "baseline", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    items: list[dict] = []
    if mode in {"donation", "full"}:
        items.extend(
            donation_grid(
                gamma_values=gamma_values,
                beta_values=beta_values,
                alpha_values=alpha_values,
                time_values=time_values,
                c_values=c_values,
            )
        )

    if mode in {"baseline", "full"}:
        items.extend(
            baseline_grid(
                gamma_values=gamma_values,
                beta_values=beta_values,
                alpha_values=alpha_values,
                time_values=time_values,
                baseline_pd=baseline_pd,
            )
        )

    return items

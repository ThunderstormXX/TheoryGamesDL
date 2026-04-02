"""Plotting helpers for trap search artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .trap_detection import TrapDetectorConfig, moving_average


def _format_pd(pd: list[float]) -> str:
    return "[" + ", ".join(f"{float(x):g}" for x in pd) + "]"


def save_trap_plot(
    sim_result: dict,
    trap_report: dict,
    output_path: Path,
    detector_cfg: TrapDetectorConfig,
    title_prefix: str = "exp10 trap search",
) -> None:
    """Save a two-panel plot with probabilities and Q-values."""
    p1 = np.asarray(sim_result["prob_c_player1"], dtype=float)
    p2 = np.asarray(sim_result["prob_c_player2"], dtype=float)
    q1c = np.asarray(sim_result["Q-val_player1_act_c"], dtype=float)
    q1d = np.asarray(sim_result["Q-val_player1_act_d"], dtype=float)
    q2c = np.asarray(sim_result["Q-val_player2_act_c"], dtype=float)
    q2d = np.asarray(sim_result["Q-val_player2_act_d"], dtype=float)

    meta = sim_result.get("meta", {})
    pd = meta.get("pd", [])

    n = len(p1)
    smooth_w = detector_cfg.smooth_window
    if n > 2:
        smooth_w = min(smooth_w, n if n % 2 == 1 else n - 1)
        if smooth_w < 1:
            smooth_w = 1
        if smooth_w % 2 == 0:
            smooth_w -= 1
    else:
        smooth_w = 1

    p1_s = moving_average(p1, smooth_w)
    p2_s = moving_average(p2, smooth_w)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax0 = axes[0]
    ax0.plot(p1, color="#6fa8dc", alpha=0.35, linewidth=1, label="p1 raw")
    ax0.plot(p2, color="#f6b26b", alpha=0.35, linewidth=1, label="p2 raw")
    ax0.plot(p1_s, color="#0b5394", linewidth=2, label=f"p1 smooth (w={smooth_w})")
    ax0.plot(p2_s, color="#e69138", linewidth=2, label=f"p2 smooth (w={smooth_w})")

    p1_jump = trap_report.get("player1", {}).get("jump_idx")
    p2_jump = trap_report.get("player2", {}).get("jump_idx")
    best_jump = trap_report.get("jump_idx")
    best_player = trap_report.get("player")

    if p1_jump is not None:
        ax0.axvline(p1_jump, color="#0b5394", linestyle="--", alpha=0.6)
    if p2_jump is not None:
        ax0.axvline(p2_jump, color="#e69138", linestyle="--", alpha=0.6)
    if best_jump is not None:
        ax0.axvline(best_jump, color="black", linestyle="-", alpha=0.8, linewidth=1.5)

    ax0.axhline(detector_cfg.near_zero_thr, color="gray", linestyle=":", linewidth=1)
    ax0.axhline(detector_cfg.high_thr, color="gray", linestyle="--", linewidth=1)
    ax0.set_ylabel("P(cooperate)")
    ax0.set_title("Cooperation trajectories")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", ncol=2)

    ax1 = axes[1]
    ax1.plot(q1c, color="#1b9e77", linewidth=1.8, label="Q1(C)")
    ax1.plot(q1d, color="#d95f02", linewidth=1.8, label="Q1(D)")
    ax1.plot(q2c, color="#7570b3", linewidth=1.4, label="Q2(C)")
    ax1.plot(q2d, color="#e7298a", linewidth=1.4, label="Q2(D)")
    if best_jump is not None:
        ax1.axvline(best_jump, color="black", linestyle="-", alpha=0.8, linewidth=1.5)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Q-value")
    ax1.set_title("Q trajectories")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", ncol=2)

    subtitle = (
        f"pd={_format_pd(pd)}"
        f", B={meta.get('B')}, C={meta.get('C')}, alpha={meta.get('alpha')}, beta={meta.get('beta')}, "
        f"gamma={meta.get('gamma')}, time={meta.get('time')}, score={trap_report.get('score', 0.0):.4f}, "
        f"is_trap={trap_report.get('is_trap')}, player={best_player}, jump_idx={best_jump}"
    )
    fig.suptitle(f"{title_prefix}\n{subtitle}", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

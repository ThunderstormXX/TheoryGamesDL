"""Transparent trap detector for cooperation probability trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrapDetectorConfig:
    near_zero_thr: float = 0.05
    high_thr: float = 0.20
    min_low_len_frac: float = 0.10
    jump_window: int = 2000
    min_jump: float = 0.10
    post_stable_frac: float = 0.20
    rel_drop_tol: float = 0.5
    smooth_window: int = 501
    early_search_frac: float = 0.50
    min_post_above_frac: float = 0.70
    tail_frac: float = 0.10
    w_low: float = 0.30
    w_jump: float = 0.30
    w_stable: float = 0.25
    w_tail_gain: float = 0.15


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _adaptive_window(window: int, n: int) -> int:
    if n <= 2:
        return 1
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w = max(1, w - 1)
    return w


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average preserving length with edge padding."""
    if window <= 1:
        return x.astype(float, copy=True)

    pad = window // 2
    padded = np.pad(x, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) for True runs."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def detect_trap_single_player(
    p: list[float] | np.ndarray,
    cfg: TrapDetectorConfig,
) -> dict:
    """Detect trap pattern on one player's cooperation trajectory."""
    p_raw = np.asarray(p, dtype=float)
    if p_raw.size == 0:
        return {
            "is_trap": False,
            "score": 0.0,
            "reason": "empty_series",
        }

    p_raw = np.clip(p_raw, 0.0, 1.0)
    n = int(p_raw.size)

    smooth_w = _adaptive_window(cfg.smooth_window, n)
    p_smooth = moving_average(p_raw, smooth_w)

    low_mask = p_smooth <= cfg.near_zero_thr
    low_runs = _find_runs(low_mask)
    early_limit = max(1, int(cfg.early_search_frac * n))

    early_runs = [run for run in low_runs if run[0] < early_limit]
    if early_runs:
        low_start, low_end = max(early_runs, key=lambda r: (r[1] - r[0], -r[0]))
    else:
        low_start, low_end = (0, 0)

    low_len = int(low_end - low_start)
    low_len_frac = (low_len / n) if n > 0 else 0.0
    min_low_len = max(1, int(math.ceil(cfg.min_low_len_frac * n)))
    low_ok = low_len >= min_low_len

    jump_window = int(max(1, min(cfg.jump_window, n // 2 if n > 2 else 1)))
    jump_search_start = low_end

    jump_ok = False
    jump_idx = None
    jump_base_idx = None
    p_before = 0.0
    p_after = 0.0
    jump_size = 0.0

    if jump_search_start + jump_window < n:
        deltas = p_smooth[jump_search_start + jump_window :] - p_smooth[jump_search_start : n - jump_window]
        local_idx = int(np.argmax(deltas))
        jump_base_idx = int(jump_search_start + local_idx)
        jump_idx = int(jump_base_idx + jump_window)
        p_before = float(p_smooth[jump_base_idx])
        p_after = float(p_smooth[jump_idx])
        jump_size = float(p_after - p_before)
        jump_ok = jump_size >= cfg.min_jump and p_after >= cfg.high_thr

    post_len = 0
    post_mean = 0.0
    post_min = 0.0
    post_above_frac = 0.0
    post_len_ok = False
    post_min_ok = False
    post_above_ok = False
    tail_mean_gain = 0.0
    tail_gain_ok = False

    if jump_idx is not None and jump_idx < n:
        post = p_smooth[jump_idx:]
        post_len = int(post.size)
        post_mean = _safe_mean(post)
        post_min = float(np.min(post)) if post.size else 0.0
        post_above_frac = float(np.mean(post > cfg.near_zero_thr)) if post.size else 0.0

        min_post_len = max(1, int(math.ceil(cfg.post_stable_frac * n)))
        post_len_ok = post_len >= min_post_len

        near_zero_floor = cfg.near_zero_thr * cfg.rel_drop_tol
        post_min_ok = post_min > near_zero_floor
        post_above_ok = post_above_frac >= cfg.min_post_above_frac

        tail_n = max(5, int(cfg.tail_frac * n))
        tail_n = min(tail_n, n)

        pre_start = max(0, jump_base_idx - tail_n + 1) if jump_base_idx is not None else 0
        pre_slice = p_smooth[pre_start : (jump_base_idx + 1 if jump_base_idx is not None else 1)]
        pre_mean = _safe_mean(pre_slice)

        tail_slice = p_smooth[-tail_n:]
        tail_mean = _safe_mean(tail_slice)

        tail_mean_gain = float(tail_mean - pre_mean)
        tail_gain_ok = tail_mean_gain > 0 and post_mean > pre_mean
    else:
        pre_mean = _safe_mean(p_smooth[: max(1, n // 10)])
        near_zero_floor = cfg.near_zero_thr * cfg.rel_drop_tol

    low_norm = _clip01((low_len_frac - cfg.min_low_len_frac) / max(1e-9, 0.5 - cfg.min_low_len_frac))
    jump_norm = _clip01((jump_size - cfg.min_jump) / max(1e-9, 0.5 - cfg.min_jump))
    stability_norm = _clip01(
        0.7 * post_above_frac
        + 0.3 * ((post_min - near_zero_floor) / max(1e-9, cfg.high_thr - near_zero_floor))
    )
    tail_norm = _clip01(tail_mean_gain / max(cfg.min_jump, 1e-9))

    raw_score = (
        cfg.w_low * low_norm
        + cfg.w_jump * jump_norm
        + cfg.w_stable * stability_norm
        + cfg.w_tail_gain * tail_norm
    )

    checks = [low_ok, jump_ok, post_len_ok, post_min_ok, post_above_ok, tail_gain_ok]
    passed = int(sum(checks))
    score = float(raw_score if all(checks) else raw_score * (0.7 * passed / len(checks)))

    is_trap = bool(all(checks))

    return {
        "is_trap": is_trap,
        "score": score,
        "low_segment_start": int(low_start),
        "low_segment_end": int(low_end),
        "low_segment_len": low_len,
        "low_segment_len_frac": low_len_frac,
        "jump_idx": jump_idx,
        "jump_base_idx": jump_base_idx,
        "jump_window_used": jump_window,
        "jump_size": jump_size,
        "pre_jump_mean": float(pre_mean),
        "post_jump_mean": float(post_mean),
        "post_jump_min": float(post_min),
        "post_above_near_zero_frac": float(post_above_frac),
        "post_len": int(post_len),
        "smooth_window_used": int(smooth_w),
        "criteria": {
            "low_ok": bool(low_ok),
            "jump_ok": bool(jump_ok),
            "post_len_ok": bool(post_len_ok),
            "post_min_ok": bool(post_min_ok),
            "post_above_ok": bool(post_above_ok),
            "tail_gain_ok": bool(tail_gain_ok),
        },
        "components": {
            "low_norm": float(low_norm),
            "jump_norm": float(jump_norm),
            "stability_norm": float(stability_norm),
            "tail_gain_norm": float(tail_norm),
        },
        "tail_mean_gain": float(tail_mean_gain),
    }


def detect_trap_for_players(
    prob_c_player1: list[float] | np.ndarray,
    prob_c_player2: list[float] | np.ndarray,
    cfg: TrapDetectorConfig,
) -> dict:
    """Run detector for both players and select strongest trap report."""
    p1 = detect_trap_single_player(prob_c_player1, cfg)
    p2 = detect_trap_single_player(prob_c_player2, cfg)

    traps = []
    if p1.get("is_trap", False):
        traps.append((1, p1))
    if p2.get("is_trap", False):
        traps.append((2, p2))

    if traps:
        player, best = max(traps, key=lambda item: item[1]["score"])
        is_trap = True
    else:
        player, best = (None, p1 if p1.get("score", 0.0) >= p2.get("score", 0.0) else p2)
        is_trap = False

    return {
        "is_trap": bool(is_trap),
        "player": player,
        "jump_idx": best.get("jump_idx"),
        "low_segment_len": best.get("low_segment_len"),
        "pre_jump_mean": best.get("pre_jump_mean"),
        "post_jump_mean": best.get("post_jump_mean"),
        "post_jump_min": best.get("post_jump_min"),
        "jump_size": best.get("jump_size"),
        "score": float(best.get("score", 0.0)),
        "player1": p1,
        "player2": p2,
        "detector_config": asdict(cfg),
    }

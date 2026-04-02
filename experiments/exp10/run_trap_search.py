#!/usr/bin/env python3
"""exp10: automatic trap search for Boltzmann Q-learning in 2x2 social dilemmas."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Ensure local imports work when launching as script from repo root.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils.io_utils import (  # noqa: E402
    append_csv_row,
    append_jsonl,
    ensure_dir,
    existing_run_keys,
    flatten_for_csv,
    load_jsonl,
    make_run_key,
    parse_bool,
    pd_to_str,
    sort_rows_by_score,
    write_json,
)
from utils.param_grid import (  # noqa: E402
    BASELINE_PD,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_C,
    DEFAULT_GAMMA,
    DEFAULT_TIME,
    build_grid,
)
from utils.plotting import save_trap_plot  # noqa: E402
from utils.simulate_wrapper import SimConfig, run_simulation  # noqa: E402
from utils.trap_detection import TrapDetectorConfig, detect_trap_for_players  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="exp10 trap search")
    parser.add_argument("--mode", choices=["donation", "baseline", "full"], default="donation")

    parser.add_argument("--gamma", type=float, nargs="+", default=DEFAULT_GAMMA)
    parser.add_argument("--beta", type=float, nargs="+", default=DEFAULT_BETA)
    parser.add_argument("--alpha", type=float, nargs="+", default=DEFAULT_ALPHA)
    parser.add_argument("--time", type=int, nargs="+", default=DEFAULT_TIME)
    parser.add_argument("--C", type=float, nargs="+", default=DEFAULT_C)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=None, help="Hard cap for number of pending runs")

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--plot-best-only", action="store_true")

    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results")
    parser.add_argument("--artifacts-dir", type=Path, default=THIS_DIR / "artifacts")

    parser.add_argument("--near-zero-thr", type=float, default=0.05)
    parser.add_argument("--high-thr", type=float, default=0.20)
    parser.add_argument("--min-low-len-frac", type=float, default=0.10)
    parser.add_argument("--jump-window", type=int, default=2000)
    parser.add_argument("--min-jump", type=float, default=0.10)
    parser.add_argument("--post-stable-frac", type=float, default=0.20)
    parser.add_argument("--rel-drop-tol", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=501)

    return parser


def _detector_cfg_from_args(args: argparse.Namespace) -> TrapDetectorConfig:
    return TrapDetectorConfig(
        near_zero_thr=args.near_zero_thr,
        high_thr=args.high_thr,
        min_low_len_frac=args.min_low_len_frac,
        jump_window=args.jump_window,
        min_jump=args.min_jump,
        post_stable_frac=args.post_stable_frac,
        rel_drop_tol=args.rel_drop_tol,
        smooth_window=args.smooth_window,
    )


def _prepare_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = build_grid(
        mode=args.mode,
        gamma_values=args.gamma,
        beta_values=args.beta,
        alpha_values=args.alpha,
        time_values=args.time,
        c_values=args.C,
        baseline_pd=BASELINE_PD,
    )

    tasks: list[dict[str, Any]] = []
    for idx, cfg in enumerate(grid):
        seed = int(args.seed + idx)
        run_key = make_run_key(
            pd=cfg["pd"],
            gamma=cfg["gamma"],
            beta=cfg["beta"],
            alpha=cfg["alpha"],
            time=cfg["time"],
            seed=seed,
            grid_type=cfg["grid_type"],
        )
        task = {
            "run_idx": idx,
            "run_key": run_key,
            "seed": seed,
            "mode": args.mode,
            **cfg,
        }
        tasks.append(task)
    return tasks


def _run_task(task: dict[str, Any], detector_cfg: TrapDetectorConfig) -> dict[str, Any]:
    sim_cfg = SimConfig(
        pd=task["pd"],
        time=int(task["time"]),
        gamma=float(task["gamma"]),
        alpha=float(task["alpha"]),
        beta=float(task["beta"]),
        seed=int(task["seed"]),
        mode=str(task["mode"]),
        grid_type=str(task["grid_type"]),
        B=task.get("B"),
        C=task.get("C"),
    )

    sim_res = run_simulation(sim_cfg)
    trap = detect_trap_for_players(
        prob_c_player1=sim_res["prob_c_player1"],
        prob_c_player2=sim_res["prob_c_player2"],
        cfg=detector_cfg,
    )

    meta = sim_res["meta"]
    row = {
        "run_idx": int(task["run_idx"]),
        "run_key": task["run_key"],
        "mode": task["mode"],
        "grid_type": task["grid_type"],
        "pd": [float(x) for x in task["pd"]],
        "pd_str": pd_to_str(task["pd"]),
        "B": task.get("B"),
        "C": task.get("C"),
        "gamma": float(task["gamma"]),
        "beta": float(task["beta"]),
        "alpha": float(task["alpha"]),
        "time": int(task["time"]),
        "seed": int(task["seed"]),
        "g1": float(meta["gaps"]["g1"]),
        "g2": float(meta["gaps"]["g2"]),
        "g3": float(meta["gaps"]["g3"]),
        "is_konstantinov": bool(meta["is_konstantinov"]),
        "trap_player": trap["player"],
        "trap_jump_idx": trap["jump_idx"],
        "trap_low_segment_len": trap["low_segment_len"],
        "trap_pre_jump_mean": trap["pre_jump_mean"],
        "trap_post_jump_mean": trap["post_jump_mean"],
        "trap_post_jump_min": trap["post_jump_min"],
        "trap_jump_size": trap["jump_size"],
        "trap_score": float(trap["score"]),
        "is_trap": bool(trap["is_trap"]),
        "player1_report": trap["player1"],
        "player2_report": trap["player2"],
        "detector_config": trap["detector_config"],
    }
    return row


def _worker_run(task: dict[str, Any], detector_cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = TrapDetectorConfig(**detector_cfg_dict)
    return _run_task(task, cfg)


def _plot_name_from_row(row: dict[str, Any]) -> str:
    key = str(row["run_key"])
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"trap_{int(row['run_idx']):05d}_{key_hash}.png"


def _rerun_for_plot(row: dict[str, Any], detector_cfg: TrapDetectorConfig, artifacts_dir: Path) -> Path:
    sim_cfg = SimConfig(
        pd=[float(x) for x in row["pd"]],
        time=int(row["time"]),
        gamma=float(row["gamma"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        seed=int(row["seed"]),
        mode=str(row["mode"]),
        grid_type=str(row["grid_type"]),
        B=row.get("B"),
        C=row.get("C"),
    )
    sim_res = run_simulation(sim_cfg)

    trap_report = {
        "is_trap": row["is_trap"],
        "score": row["trap_score"],
        "player": row["trap_player"],
        "jump_idx": row["trap_jump_idx"],
        "player1": row.get("player1_report", {}),
        "player2": row.get("player2_report", {}),
    }

    out_path = artifacts_dir / _plot_name_from_row(row)
    save_trap_plot(sim_res, trap_report, out_path, detector_cfg=detector_cfg)
    return out_path


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    ensure_dir(args.results_dir)
    ensure_dir(args.artifacts_dir)

    detector_cfg = _detector_cfg_from_args(args)

    jsonl_path = args.results_dir / "trap_search_results.jsonl"
    csv_path = args.results_dir / "trap_search_results.csv"
    best_json_path = args.results_dir / "best_traps.json"

    tasks = _prepare_tasks(args)
    existing_keys = existing_run_keys(jsonl_path) if args.resume else set()

    pending = [t for t in tasks if t["run_key"] not in existing_keys]
    if args.max_runs is not None:
        pending = pending[: max(0, args.max_runs)]

    print(
        f"Mode={args.mode}; total grid={len(tasks)}; already_done={len(existing_keys)}; "
        f"pending={len(pending)}; jobs={args.jobs}"
    )

    start = time.time()
    new_rows: list[dict[str, Any]] = []
    fieldnames_cache: list[str] | None = None

    if pending:
        if args.jobs <= 1:
            for task in tqdm(pending, desc="exp10 trap search"):
                row = _run_task(task, detector_cfg)
                append_jsonl(jsonl_path, row)
                csv_row = flatten_for_csv(row)
                if fieldnames_cache is None:
                    fieldnames_cache = list(csv_row.keys())
                append_csv_row(csv_path, csv_row, fieldnames=fieldnames_cache)
                new_rows.append(row)
        else:
            detector_cfg_dict = asdict(detector_cfg)
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futures = [ex.submit(_worker_run, task, detector_cfg_dict) for task in pending]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="exp10 trap search"):
                    row = fut.result()
                    append_jsonl(jsonl_path, row)
                    csv_row = flatten_for_csv(row)
                    if fieldnames_cache is None:
                        fieldnames_cache = list(csv_row.keys())
                    append_csv_row(csv_path, csv_row, fieldnames=fieldnames_cache)
                    new_rows.append(row)

    elapsed = time.time() - start
    print(f"Finished pending runs in {elapsed:.1f}s")

    all_rows = load_jsonl(jsonl_path)
    all_sorted = sort_rows_by_score(all_rows, traps_only=True)
    best_rows = all_sorted[: max(0, int(args.top_k))]

    write_json(best_json_path, best_rows)

    total_runs = len(all_rows)
    total_traps = sum(1 for r in all_rows if parse_bool(r.get("is_trap", False)))
    donation_traps = sum(
        1
        for r in all_rows
        if parse_bool(r.get("is_trap", False)) and parse_bool(r.get("is_konstantinov", False))
    )

    print(
        f"Summary: total_runs={total_runs}, traps={total_traps}, "
        f"donation_traps={donation_traps}, best_saved={len(best_rows)}"
    )

    if args.plot_best_only:
        rows_for_plot = best_rows
    else:
        rows_for_plot = [r for r in new_rows if parse_bool(r.get("is_trap", False))]

    plotted = 0
    if rows_for_plot:
        print(f"Generating plots: {len(rows_for_plot)}")
        for row in tqdm(rows_for_plot, desc="plotting"):
            _rerun_for_plot(row, detector_cfg, args.artifacts_dir)
            plotted += 1
    print(f"Plots saved: {plotted}")

    print("Results files:")
    print(f" - {csv_path}")
    print(f" - {jsonl_path}")
    print(f" - {best_json_path}")
    print(f"Artifacts dir: {args.artifacts_dir}")


if __name__ == "__main__":
    # MacOS often defaults to "spawn" for subprocesses; this script is spawn-safe.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()

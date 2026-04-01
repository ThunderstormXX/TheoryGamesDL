import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _get_nested(d: Dict[str, Any], path: str) -> Optional[float]:
    cur: Any = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    try:
        val = float(cur)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def _collect(
    runs: Iterable[Dict[str, Any]],
    *,
    metric_path: str,
    graphs: Tuple[str, ...] = ('edge2', 'triangle3', 'star3'),
) -> Tuple[List[float], List[float], Dict[Tuple[str, float], List[Tuple[float, float]]]]:
    """Return (betas, gammas, points) where points[(graph,gamma)] = [(beta, metric), ...]."""
    points: Dict[Tuple[str, float], List[Tuple[float, float]]] = {}
    betas_set = set()
    gammas_set = set()

    for r in runs:
        g = r.get('graph')
        if g not in graphs:
            continue
        beta = r.get('beta')
        gamma = r.get('gamma')
        if beta is None or gamma is None:
            continue
        try:
            beta_f = float(beta)
            gamma_f = float(gamma)
        except Exception:
            continue

        val = _get_nested(r, metric_path)
        if val is None:
            continue

        betas_set.add(beta_f)
        gammas_set.add(gamma_f)
        points.setdefault((g, gamma_f), []).append((beta_f, val))

    betas = sorted(betas_set)
    gammas = sorted(gammas_set)

    # Sort per-series
    for k in list(points.keys()):
        points[k] = sorted(points[k], key=lambda t: t[0])

    return betas, gammas, points


def plot_compare_n2_vs_n3(
    *,
    summary: Dict[str, Any],
    metric_path: str,
    ylabel: str,
    out_path: Path,
) -> None:
    runs = summary.get('runs')
    if not isinstance(runs, list):
        raise ValueError('summary.json has no runs[]')

    betas, gammas, points = _collect(runs, metric_path=metric_path)
    if not gammas:
        raise ValueError(f'No usable data for metric "{metric_path}" in summary runs')

    graphs = ['edge2', 'triangle3', 'star3']
    styles = {
        'edge2': dict(color='black', linestyle='-', marker='o'),
        'triangle3': dict(color='tab:blue', linestyle='--', marker='s'),
        'star3': dict(color='tab:green', linestyle=':', marker='D'),
    }

    # Layout: subplots per gamma
    n = len(gammas)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.array(axes).reshape(-1)

    benefit = summary.get('benefit')
    cost = summary.get('cost')
    reward = summary.get('reward_type')
    header_parts = []
    if reward is not None:
        header_parts.append(f"reward={reward}")
    if benefit is not None:
        header_parts.append(f"b={benefit}")
    if cost is not None:
        header_parts.append(f"c={cost}")
    header = ' | '.join(header_parts)

    fig.suptitle(f'2 vs 3 agents comparison: {ylabel}\n{header}', fontsize=12)

    for i, gamma in enumerate(gammas):
        ax = axes_arr[i]
        for g in graphs:
            series = points.get((g, gamma), [])
            if not series:
                continue
            xs = [b for b, _ in series]
            ys = [v for _, v in series]
            st = styles[g]
            ax.plot(xs, ys, label=g, linewidth=1.4, markersize=4, **st)

        ax.set_title(f'gamma={gamma:g}')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('beta (higher = less noise)')
        ax.set_ylabel(ylabel)

        if metric_path == 'trap_fraction_any_rep':
            ax.set_ylim(-0.02, 1.02)

        ax.legend(loc='best', fontsize=8)

    # Turn off extra axes
    for j in range(len(gammas), len(axes_arr)):
        axes_arr[j].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def _metric_presets() -> List[Tuple[str, str]]:
    return [
        ('trap_fraction_any_rep', 'trap_fraction_any_rep'),
        ('trap_exit_fraction_any_rep', 'trap_exit_fraction_any_rep'),
        ('trap_exit_fraction_given_trap', 'trap_exit_fraction_given_trap'),
        ('deltaq_inc_stats.std', 'std(δ(ΔQ))'),
        ('deltaq_inc_stats.excess_kurtosis', 'excess kurtosis of δ(ΔQ)'),
        ('deltaq_inc_stats.abs_mean', 'mean |δ(ΔQ)|'),
        ('volatility_acf_summary.acf_sum_1_to_K', 'sum ACF[1..K] of |δ(ΔQ)|'),
        ('volatility_acf_summary.acf1', 'ACF lag-1 of |δ(ΔQ)|'),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description='Post-hoc comparison plots (2 vs 3 agents) from trap_effect summary_*.json')
    ap.add_argument('--summary', type=str, required=True, help='Path to summary_*.json')
    ap.add_argument(
        '--metric',
        type=str,
        default='trap_fraction_any_rep',
        help='Single metric path in runs (supports dotted paths). Use --metrics/--all for multiple outputs.',
    )
    ap.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='Comma-separated metric paths to generate multiple plots. Example: trap_fraction_any_rep,deltaq_inc_stats.std,volatility_acf_summary.acf_sum_1_to_K',
    )
    ap.add_argument(
        '--all',
        action='store_true',
        help='Generate a preset bundle of comparison plots (trap, deltaQ stats, volatility clustering) if data is available.',
    )
    ap.add_argument('--ylabel', type=str, default=None, help='Y axis label (optional)')
    ap.add_argument('--out', type=str, default=None, help='Output png path (optional). Default: next to summary file.')
    args = ap.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    def _run_one(metric_path: str, ylabel: Optional[str], out_path: Optional[Path]) -> Optional[Path]:
        ylab = ylabel or metric_path
        if out_path is None:
            out_path_final = summary_path.parent / f'compare_n2_vs_n3_{metric_path.replace(".", "_")}.png'
        else:
            out_path_final = out_path
        plot_compare_n2_vs_n3(summary=summary, metric_path=metric_path, ylabel=ylab, out_path=out_path_final)
        print(str(out_path_final))
        return out_path_final

    if args.all:
        for metric_path, ylabel in _metric_presets():
            try:
                _run_one(metric_path, ylabel, None)
            except ValueError as exc:
                print(f"[skip] {metric_path}: {exc}")
        return

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
        for metric_path in metrics:
            try:
                _run_one(metric_path, None, None)
            except ValueError as exc:
                print(f"[skip] {metric_path}: {exc}")
        return

    ylabel = args.ylabel
    if ylabel is None:
        ylabel = args.metric

    if args.out is None:
        out_path = summary_path.parent / f'compare_n2_vs_n3_{args.metric.replace(".", "_")}.png'
    else:
        out_path = Path(args.out).expanduser().resolve()

    _run_one(args.metric, ylabel, out_path)


if __name__ == '__main__':
    main()

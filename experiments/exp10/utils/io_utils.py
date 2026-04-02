"""I/O helpers for exp10 results, resume and serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pd_to_str(pd: list[float]) -> str:
    return ",".join(f"{float(x):g}" for x in pd)


def make_run_key(
    pd: list[float],
    gamma: float,
    beta: float,
    alpha: float,
    time: int,
    seed: int | None,
    grid_type: str,
) -> str:
    return (
        f"{grid_type}|pd={pd_to_str(pd)}|gamma={gamma:.12g}|beta={beta:.12g}|"
        f"alpha={alpha:.12g}|time={int(time)}|seed={seed}"
    )


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def existing_run_keys(jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    for row in load_jsonl(jsonl_path):
        key = row.get("run_key")
        if isinstance(key, str):
            keys.add(key)
    return keys


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    file_exists = path.exists()
    if fieldnames is None:
        fieldnames = list(row.keys())

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sort_rows_by_score(rows: list[dict[str, Any]], traps_only: bool = True) -> list[dict[str, Any]]:
    if traps_only:
        rows = [r for r in rows if parse_bool(r.get("is_trap", False))]
    return sorted(rows, key=lambda r: as_float(r.get("trap_score", r.get("score", 0.0))), reverse=True)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_for_csv(result_row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested player detector sections for compact CSV output."""
    row = dict(result_row)

    p1 = row.pop("player1_report", None)
    p2 = row.pop("player2_report", None)

    for prefix, report in (("p1", p1), ("p2", p2)):
        if not isinstance(report, dict):
            continue
        row[f"{prefix}_is_trap"] = report.get("is_trap")
        row[f"{prefix}_score"] = report.get("score")
        row[f"{prefix}_jump_idx"] = report.get("jump_idx")
        row[f"{prefix}_jump_size"] = report.get("jump_size")
        row[f"{prefix}_low_segment_len"] = report.get("low_segment_len")
        row[f"{prefix}_pre_jump_mean"] = report.get("pre_jump_mean")
        row[f"{prefix}_post_jump_mean"] = report.get("post_jump_mean")
        row[f"{prefix}_post_jump_min"] = report.get("post_jump_min")
        row[f"{prefix}_post_above_near_zero_frac"] = report.get("post_above_near_zero_frac")
        criteria = report.get("criteria", {}) or {}
        row[f"{prefix}_criteria_low_ok"] = criteria.get("low_ok")
        row[f"{prefix}_criteria_jump_ok"] = criteria.get("jump_ok")
        row[f"{prefix}_criteria_post_len_ok"] = criteria.get("post_len_ok")
        row[f"{prefix}_criteria_post_min_ok"] = criteria.get("post_min_ok")
        row[f"{prefix}_criteria_post_above_ok"] = criteria.get("post_above_ok")
        row[f"{prefix}_criteria_tail_gain_ok"] = criteria.get("tail_gain_ok")

    return row

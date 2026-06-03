#!/usr/bin/env python3
"""Build or refresh ``all_checkpoints.json`` for rollout-vs-loss plotting.

Scans ``<run_dir>/ckpt/`` for saved Lightning checkpoints (``step*-val_l1_*.pth``,
plus optional ``last.pth``), pulls training metrics from the run's CSV log, and
merges rollout success rates from eval summaries under ``<results_dir>/ckpt_*/``.

Output schema (``points`` sorted by training step, whatever intervals exist):

    {
      "run_dir": "...",
      "run_folder": "...",
      "results_dir": "...",
      "task_id": 53,
      "task_name": "object_scaling",
      "metrics_csv": ".../logs/metrics.csv",
      "points": [
        {
          "step": 5000,
          "checkpoint_file": "step5000-val_l1_0.03541.pth",
          "checkpoint_path": ".../ckpt/step5000-val_l1_0.03541.pth",
          "val_l1": 0.03541,
          "val_loss": 0.03541,
          "train_loss": 0.0281,
          "success_rate": 0.65,
          "n_success": 13,
          "n_episodes": 20,
          "evaluated": true
        },
        ...
      ]
    }

Usage:
    python il/il_lib/build_all_checkpoints_json.py \\
        --run-dir /path/to/run_dir \\
        --results-dir /path/to/results/<run_folder>

``--results-dir`` defaults to ``results/<run_folder>`` under the repo root,
or ``results/diffusion/<run_folder>`` when the run lives under
``checkpoints/diffusion/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

STEP_VAL_L1_RE = re.compile(r"^step(\d+)-val_l1_([0-9.]+)\.pth$")
GCBC_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")

TRAIN_LOSS_COLS = ("train/loss", "train/loss_epoch", "train_loss")
VAL_LOSS_COLS = ("val/loss", "val/loss_epoch", "val_loss")
VAL_L1_COLS = ("val/l1", "val/l1_epoch", "val_l1")


def repo_root() -> Path:
    # .../behavior-1k-private/il/il_lib/build_all_checkpoints_json.py
    return Path(__file__).resolve().parent.parent.parent


def _parse_task_from_run_folder(run_folder: str) -> tuple[int | None, str | None]:
    stem = run_folder
    m_ts = re.match(r"^(.+)_[0-9]{8}-[0-9]{6}$", run_folder)
    if m_ts:
        stem = m_ts.group(1)

    task_map = {
        51: "camera_relocalization",
        53: "object_scaling",
        54: "mental_rotation",
    }

    for pat in (
        r".+-perturbed-task-0*([0-9]+)-",
        r"single-goal-image-task-0*([0-9]+)-",
        r"single-goal-task-0*([0-9]+)-",
    ):
        m = re.match(pat, stem)
        if m:
            tid = int(m.group(1))
            return tid, task_map.get(tid)
    return None, None


def _to_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _pick_col(row: dict[str, str], candidates: tuple[str, ...]) -> float | None:
    for col in candidates:
        if col in row:
            v = _to_float(row[col])
            if v is not None:
                return v
    return None


def load_metric_series(metrics_csv: Path) -> dict[str, list[tuple[int, float]]]:
    """Per-metric (step, value) series, sorted by step.

    Lightning's CSVLogger writes train and val metrics on separate rows that
    can share the same ``step`` (e.g. an lr/train row and a ``val/*`` row both
    at step 1999), and leaves the other columns blank. Indexing the whole row
    by step and taking the last row would drop ``train/loss`` whenever a blank
    val row follows it. So we collect each metric independently, keeping only
    the rows where that metric is actually present.
    """
    series: dict[str, list[tuple[int, float]]] = {
        "train_loss": [],
        "val_loss": [],
        "val_l1": [],
    }
    if not metrics_csv.is_file():
        return series

    cols_by_key = {
        "train_loss": TRAIN_LOSS_COLS,
        "val_loss": VAL_LOSS_COLS,
        "val_l1": VAL_L1_COLS,
    }
    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = _to_float(row.get("step"))
            if step is None:
                continue
            step_i = int(step)
            for key, cols in cols_by_key.items():
                v = _pick_col(row, cols)
                if v is not None:
                    series[key].append((step_i, v))

    for key in series:
        series[key].sort(key=lambda sv: sv[0])
    return series


def lookup_metric(series: list[tuple[int, float]], step: int) -> float | None:
    """Value at the largest logged step <= ``step`` (else None)."""
    best: float | None = None
    for s, v in series:
        if s <= step:
            best = v
        else:
            break
    return best


def discover_checkpoints(ckpt_dir: Path) -> list[tuple[int, str, float | None]]:
    """Return [(step, filename, val_l1_from_name), ...] sorted by step."""
    found: list[tuple[int, str, float | None]] = []
    has_last = False

    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file() or p.suffix != ".pth":
            continue
        name = p.name
        if "-train_loss" in name or "-val_loss" in name or name == "last-v1.pth":
            continue
        m = STEP_VAL_L1_RE.match(name)
        if m:
            found.append((int(m.group(1)), name, float(m.group(2))))
            continue
        if name == "last.pth":
            has_last = True

    found.sort(key=lambda x: x[0])

    # last.pth usually duplicates the final numbered checkpoint; omit from the
    # sweep plot series when step checkpoints exist.
    if has_last and not found:
        found.append((0, "last.pth", None))

    return found


def discover_checkpoints_gcbc(ckpt_dir: Path) -> list[tuple[int, str, float | None]]:
    """GCBC layout: checkpoint_<step>.pt files directly in the run dir.

    Returns [(step, filename, None), ...] sorted by step (no val_l1 in name).
    """
    found: list[tuple[int, str, float | None]] = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file() or p.suffix != ".pt":
            continue
        m = GCBC_CKPT_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p.name, None))
    found.sort(key=lambda x: x[0])
    return found


def load_eval_summary(results_dir: Path, step: int, is_last: bool) -> dict[str, Any]:
    ckpt_key = "last" if is_last else str(step)
    summary_path = results_dir / f"ckpt_{ckpt_key}" / "summary.json"
    if not summary_path.is_file():
        return {"evaluated": False}

    with summary_path.open() as f:
        data = json.load(f)
    return {
        "evaluated": True,
        "success_rate": data.get("success_rate"),
        "n_success": data.get("n_success"),
        "n_episodes": data.get("n_episodes"),
        "mean_episode_steps": data.get("mean_episode_steps"),
    }


def infer_results_dir(run_dir: Path, project_root: Path) -> Path:
    run_folder = run_dir.name
    if "checkpoints/diffusion" in str(run_dir):
        return project_root / "results" / "diffusion" / run_folder
    return project_root / "results" / run_folder


def detect_ckpt_format(run_dir: Path) -> str:
    """Return 'lightning' if run_dir/ckpt exists, else 'gcbc' if checkpoint_*.pt."""
    if (run_dir / "ckpt").is_dir():
        return "lightning"
    if any(GCBC_CKPT_RE.match(p.name) for p in run_dir.iterdir() if p.is_file()):
        return "gcbc"
    return "lightning"


def build_payload(
    run_dir: Path,
    results_dir: Path,
    task_id: int | None = None,
    task_name: str | None = None,
    ckpt_format: str = "auto",
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    results_dir = results_dir.resolve()

    if ckpt_format == "auto":
        ckpt_format = detect_ckpt_format(run_dir)

    if ckpt_format == "gcbc":
        ckpt_dir = run_dir
        checkpoints = discover_checkpoints_gcbc(ckpt_dir)
    else:
        ckpt_dir = run_dir / "ckpt"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Missing ckpt dir: {ckpt_dir}")
        checkpoints = discover_checkpoints(ckpt_dir)

    run_folder = run_dir.name
    if task_id is None or task_name is None:
        parsed_id, parsed_name = _parse_task_from_run_folder(run_folder)
        task_id = task_id if task_id is not None else parsed_id
        task_name = task_name if task_name is not None else parsed_name

    metrics_csv = run_dir / "logs" / "metrics.csv"
    metric_series = load_metric_series(metrics_csv)

    points: list[dict[str, Any]] = []
    for step, filename, val_l1_name in checkpoints:
        is_last = filename == "last.pth"
        train_loss = lookup_metric(metric_series["train_loss"], step)
        csv_val_loss = lookup_metric(metric_series["val_loss"], step)
        csv_val_l1 = lookup_metric(metric_series["val_l1"], step)

        val_l1 = val_l1_name if val_l1_name is not None else csv_val_l1
        val_loss = csv_val_loss if csv_val_loss is not None else val_l1

        eval_data = load_eval_summary(results_dir, step, is_last=is_last)
        point: dict[str, Any] = {
            "step": step,
            "checkpoint_file": filename,
            "checkpoint_path": str(ckpt_dir / filename),
            "val_l1": val_l1,
            "val_loss": val_loss,
            "train_loss": train_loss,
            **eval_data,
        }
        points.append(point)

    return {
        "run_dir": str(run_dir),
        "run_folder": run_folder,
        "results_dir": str(results_dir),
        "ckpt_format": ckpt_format,
        "task_id": task_id,
        "task_name": task_name,
        "metrics_csv": str(metrics_csv) if metrics_csv.is_file() else None,
        "n_checkpoints": len(points),
        "n_evaluated": sum(1 for p in points if p.get("evaluated")),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Lightning run directory (contains conf.yaml and ckpt/)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Eval results root for this run (defaults based on run path)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <results-dir>/all_checkpoints.json)",
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--ckpt-format",
        choices=("auto", "lightning", "gcbc"),
        default="auto",
        help="Checkpoint layout: 'lightning' (run_dir/ckpt/step*-val_l1_*.pth), "
        "'gcbc' (run_dir/checkpoint_*.pt), or 'auto' to detect.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.name == "ckpt":
        run_dir = run_dir.parent

    project_root = repo_root()
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else infer_results_dir(run_dir, project_root)
    )
    output_path = Path(args.output) if args.output else results_dir / "all_checkpoints.json"

    payload = build_payload(
        run_dir=run_dir,
        results_dir=results_dir,
        task_id=args.task_id,
        task_name=args.task_name,
        ckpt_format=args.ckpt_format,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    n_eval = payload["n_evaluated"]
    n_total = payload["n_checkpoints"]
    print(f"Wrote {output_path} ({n_total} checkpoints, {n_eval} with rollout eval)")


if __name__ == "__main__":
    main()

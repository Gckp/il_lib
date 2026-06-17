#!/usr/bin/env python3
"""Export a wandb run's loss history to ``<run_dir>/logs/metrics.csv``.

GCBC-torch training (il/bridge_data_v2/gcbc_torch/train.py) logs losses to
wandb only -- ``training/<k>`` and ``validation/<k>`` -- and never writes them
into the .pt checkpoints or a CSV. ``build_all_checkpoints_json.py`` reads
train/val loss from ``<run_dir>/logs/metrics.csv``; this script produces that
CSV from wandb so the GCBC sweep plot gets loss curves too.

The output CSV uses the same column names the builder expects:
    step,train/loss,val/loss,val/l1

Metric mapping (wandb key -> CSV column), auto-detected from the run history:
    train/loss <- first present of training/loss, training/actor_loss, training/mse
    val/loss   <- first present of validation/loss, validation/actor_loss, validation/mse
    val/l1     <- validation/l1 (if present)
Override any of these with --train-key / --val-key / --val-l1-key.

Identifying the run:
    * --wandb-run entity/project/run_id  (most explicit), OR
    * derived from <run_dir>/train_config.json (run_name + wandb_project),
      matched by display name in the project (newest match wins).

Usage:
    python il/il_lib/fetch_wandb_metrics.py --run-dir <CHECKPOINT_DIR>
    python il/il_lib/fetch_wandb_metrics.py --run-dir <DIR> --wandb-run me/gcbc-ispatialgym/abc123

Then rebuild + plot:
    python il/il_lib/build_all_checkpoints_json.py --run-dir <DIR> \\
        --results-dir results/<RUN_FOLDER> --ckpt-format gcbc
    python il/il_lib/plot_rollout_vs_loss.py results/<RUN_FOLDER>/all_checkpoints.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

TRAIN_KEY_CANDIDATES = ("training/loss", "training/actor_loss", "training/mse")
VAL_KEY_CANDIDATES = ("validation/loss", "validation/actor_loss", "validation/mse")
VAL_L1_KEY_CANDIDATES = ("validation/l1",)


def load_train_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def resolve_run(api, args, run_dir: Path):
    if args.wandb_run:
        return api.run(args.wandb_run)

    cfg = load_train_config(run_dir)
    project = args.project or cfg.get("wandb_project")
    run_name = args.run_name or cfg.get("run_name")
    if not project or not run_name:
        raise SystemExit(
            "Could not determine wandb project/run_name. Pass --wandb-run "
            "entity/project/run_id, or --project/--run-name."
        )
    entity = args.entity or api.default_entity
    path = f"{entity}/{project}" if entity else project
    matches = list(api.runs(path, filters={"display_name": run_name}))
    if not matches:
        raise SystemExit(
            f"No wandb run named {run_name!r} found in {path}. "
            "Pass --wandb-run explicitly."
        )
    if len(matches) > 1:
        print(
            f"Warning: {len(matches)} runs named {run_name!r}; using newest.",
            file=sys.stderr,
        )
        matches.sort(key=lambda r: getattr(r, "created_at", ""), reverse=True)
    return matches[0]


def pick_key(present: set[str], override: str | None, candidates) -> str | None:
    if override:
        return override
    for c in candidates:
        if c in present:
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="GCBC checkpoint dir (has train_config.json)")
    parser.add_argument("--wandb-run", default=None, help="entity/project/run_id (overrides discovery)")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--train-key", default=None, help="wandb key for train/loss column")
    parser.add_argument("--val-key", default=None, help="wandb key for val/loss column")
    parser.add_argument("--val-l1-key", default=None, help="wandb key for val/l1 column")
    parser.add_argument("--output", default=None, help="CSV path (default <run-dir>/logs/metrics.csv)")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed in this env. `pip install wandb` and retry.")

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output) if args.output else run_dir / "logs" / "metrics.csv"

    api = wandb.Api()
    run = resolve_run(api, args, run_dir)
    print(f"Fetching history from wandb run: {run.entity}/{run.project}/{run.id} ({run.name})")

    rows = list(run.scan_history())
    if not rows:
        raise SystemExit("wandb run has no logged history.")

    present: set[str] = set()
    for r in rows:
        present.update(k for k, v in r.items() if v is not None)

    train_key = pick_key(present, args.train_key, TRAIN_KEY_CANDIDATES)
    val_key = pick_key(present, args.val_key, VAL_KEY_CANDIDATES)
    val_l1_key = pick_key(present, args.val_l1_key, VAL_L1_KEY_CANDIDATES)

    if not train_key and not val_key:
        raise SystemExit(
            "Could not find any train/val loss keys in the run history. "
            f"Available keys include: {sorted(k for k in present if '/' in k)[:30]}"
        )
    print(f"  train/loss <- {train_key or '(none)'}")
    print(f"  val/loss   <- {val_key or '(none)'}")
    print(f"  val/l1     <- {val_l1_key or '(none)'}")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train/loss", "val/loss", "val/l1"])
        for r in rows:
            step = r.get("_step")
            if step is None:
                continue
            tl = r.get(train_key) if train_key else None
            vl = r.get(val_key) if val_key else None
            v1 = r.get(val_l1_key) if val_l1_key else None
            if tl is None and vl is None and v1 is None:
                continue
            writer.writerow([
                int(step),
                "" if tl is None else tl,
                "" if vl is None else vl,
                "" if v1 is None else v1,
            ])
            n_written += 1

    print(f"Wrote {output} ({n_written} rows)")


if __name__ == "__main__":
    main()

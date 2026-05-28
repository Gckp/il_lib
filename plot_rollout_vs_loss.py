#!/usr/bin/env python3
"""Plot rollout success rate and training/validation loss vs training step.

Reads ``all_checkpoints.json`` produced by ``build_all_checkpoints_json.py``.

Example:
    python il/il_lib/plot_rollout_vs_loss.py \\
        results/d0-bg-perturbed-task-0053-.../all_checkpoints.json \\
        --output results/d0-bg-perturbed-task-0053-.../rollout_vs_loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_points(path: Path) -> tuple[dict, list[dict]]:
    with path.open() as f:
        data = json.load(f)
    points = sorted(data.get("points", []), key=lambda p: p["step"])
    return data, points


def plot_rollout_vs_loss(
    meta: dict,
    points: list[dict],
    output: Path,
    show: bool = False,
) -> None:
    steps = [p["step"] for p in points]
    success = [p.get("success_rate") for p in points]
    train_loss = [p.get("train_loss") for p in points]
    val_loss = [p.get("val_loss") for p in points]
    val_l1 = [p.get("val_l1") for p in points]

    has_success = any(s is not None for s in success)
    has_train = any(v is not None for v in train_loss)
    has_val = any(v is not None for v in val_loss)
    has_val_l1 = any(v is not None for v in val_l1)

    fig, ax_loss = plt.subplots(figsize=(10, 5))

    if has_train:
        ax_loss.plot(
            steps,
            train_loss,
            marker="o",
            linewidth=1.5,
            label="train loss",
            color="#1f77b4",
        )
    if has_val:
        ax_loss.plot(
            steps,
            val_loss,
            marker="s",
            linewidth=1.5,
            label="val loss",
            color="#ff7f0e",
        )
    elif has_val_l1:
        ax_loss.plot(
            steps,
            val_l1,
            marker="s",
            linewidth=1.5,
            label="val l1",
            color="#ff7f0e",
        )

    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    title_parts = [meta.get("run_folder", "checkpoint sweep")]
    if meta.get("task_name"):
        title_parts.append(str(meta["task_name"]))
    ax_loss.set_title(" / ".join(title_parts))

    lines, labels = ax_loss.get_legend_handles_labels()

    if has_success:
        ax_success = ax_loss.twinx()
        ax_success.plot(
            steps,
            success,
            marker="D",
            linewidth=2.0,
            label="rollout success rate",
            color="#2ca02c",
        )
        ax_success.set_ylabel("rollout success rate")
        ax_success.set_ylim(-0.05, 1.05)
        s_lines, s_labels = ax_success.get_legend_handles_labels()
        lines += s_lines
        labels += s_labels

    if lines:
        ax_loss.legend(lines, labels, loc="best")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json",
        help="Path to all_checkpoints.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <input_dir>/rollout_vs_loss.png)",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "rollout_vs_loss.png"
    )

    meta, points = load_points(input_path)
    if not points:
        raise SystemExit(f"No points found in {input_path}")

    plot_rollout_vs_loss(meta, points, output_path, show=args.show)


if __name__ == "__main__":
    main()

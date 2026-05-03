"""Prepare iSpatialGym demos for the il_lib ACT + goal-image pipeline.

Expects the iSpatialGym demo collector to write output into per-seed directories:

    datasets/
        ispatialgym-demos{suffix}-seed{N}/
            data/task-XXXX/episode_<long_name>.parquet
            videos/task-XXXX/observation.images.<mod>.<cam>/episode_<long_name>.mp4
            meta/ (optional)
        ispatialgym-instances{suffix}-seed{N}/
            task-XXXX/<instance>/
                bddl/*.bddl
                *_template.json
                *-tro_state.json
                reference_image2.png                 (the goal image)

``il_lib``'s :class:`BehaviorIterableDataset` instead expects a single,
consolidated directory with the ``2025-challenge-demos`` layout and **8-digit
episode keys** encoded as ``{task_id:04d}{demo_idx:04d}`` (see the
``int(demo_key) // 10000`` task-id math in
``OmniGibson/omnigibson/learning/datas/iterable_dataset.py``). The iSpatialGym
generator uses 14- or 16-digit names, which break that convention.

This script walks the scattered seed directories, filters by
seeds / task_ids / split, renumbers each episode to an 8-digit key, and
places (by default: symlinks) the parquet + matching videos into an output
directory that ``train.py --config-name base_config_goal_image`` can consume
directly.

It also emits:

  - ``episode_ids_split.json``: ``{"train": [...], "val": [...], "test": [...]}``
    keyed by the **new** 8-digit episode IDs.
  - ``episode_key_map.json``: maps each new key back to ``source_parquet``,
    ``source_seed``, ``source_key`` for auditability.
  - ``test_episodes.json``: for every held-out test episode, the absolute
    paths that ``eval_ispatialgym*.py`` needs (``bddl_path``,
    ``template_path``, ``tro_path``, ``goal_image_path``, renumbered
    parquet path).

Usage:

    # multi-seed
    python prepare_data.py \\
        --datasets_root datasets \\
        --demos_prefix ispatialgym-demos \\
        --instances_prefix ispatialgym-instances \\
        --seed_separator "-" \\
        --seeds 1-10 --task_ids 51-54 \\
        --output_dir datasets/act-goal-v1 \\
        --project_root /abs/path/to/behavior-1k-private \\
        --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

    # single source dir (GCBC-style)
    python prepare_data.py \\
        --data_dir datasets/ispatialgym-demos-seed-1/data/task-0051 \\
        --output_dir datasets/act-goal-v1 \\
        --project_root /abs/path/to/behavior-1k-private

    # with a user-supplied split
    python prepare_data.py ... --split_json path/to/split.json

Then train with:

    python train.py --config-name base_config_goal_image \\
        arch=act_goal task=<task_name> robot=r1pro \\
        data_dir=<abs>/datasets/act-goal-v1 \\
        goal_image_project_root=/abs/path/to/behavior-1k-private
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Small arg-parsing helpers (match convert_to_tfrecord.py where possible)
# ---------------------------------------------------------------------------


def parse_int_list(s: str) -> List[int]:
    """Parse '1-20' (inclusive) or '1,2,4,5' into a list of ints."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def resolve_instance_metadata(
    parquet_path: str, project_root: Optional[str]
) -> Tuple[Dict[str, str], List[str]]:
    """Reuse of GCBC's logic: derive bddl/template/tro/goal paths from a parquet.

    Parquet rows carry an ``image_condition_path`` pointing into the
    ``ispatialgym-instances-*`` directory; the three companion files live
    alongside it. Unlike GCBC's strict assert-based version this one is
    best-effort: it returns ``(meta_dict, warnings)`` so the caller can still
    emit a partial entry even when the instance dir is unreachable or the
    stored ``image_condition_path`` was recorded with an absolute path from a
    different machine.
    """
    warnings: List[str] = []
    df = pd.read_parquet(parquet_path, columns=["image_condition_path"])
    goal_img_path = df["image_condition_path"].iloc[0]
    if isinstance(goal_img_path, bytes):
        goal_img_path = goal_img_path.decode("utf-8")
    raw_goal_path = goal_img_path
    if not os.path.isabs(goal_img_path) and project_root:
        goal_img_path = os.path.join(project_root, goal_img_path)
    goal_img_path = os.path.abspath(goal_img_path)

    meta: Dict[str, str] = {
        "goal_image_path": goal_img_path,
        "image_condition_path_raw": raw_goal_path,
    }

    instance_dir = os.path.dirname(goal_img_path)
    if not os.path.isdir(instance_dir):
        warnings.append(
            f"instance dir missing for {parquet_path}: {instance_dir} "
            f"(raw image_condition_path={raw_goal_path!r}). Eval metadata "
            "will be incomplete; rebase the path with --project_root or copy "
            "the instance directory into the expected location."
        )
        return meta, warnings

    def _one(pattern: str, label: str) -> Optional[str]:
        hits = glob.glob(os.path.join(instance_dir, pattern))
        if len(hits) == 1:
            return os.path.abspath(hits[0])
        warnings.append(
            f"expected 1 {label} matching {pattern!r} in {instance_dir}, "
            f"got {len(hits)}"
        )
        return None

    bddl = _one(os.path.join("bddl", "*.bddl"), "BDDL file")
    template = _one("*_template.json", "template file")
    tro = _one("*-tro_state.json", "tro_state file")
    if bddl:
        meta["bddl_path"] = bddl
    if template:
        meta["template_path"] = template
    if tro:
        meta["tro_path"] = tro
    return meta, warnings


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------


def _extract_task_id_from_parquet(parquet_path: str) -> int:
    """Parse task_id from ``.../task-XXXX/episode_*.parquet``."""
    task_dir = os.path.basename(os.path.dirname(parquet_path))
    assert task_dir.startswith("task-"), f"Expected task-XXXX parent, got {task_dir!r}"
    return int(task_dir.split("-", 1)[1])


def _find_video_root_for_parquet(parquet_path: str) -> Optional[str]:
    """Given ``<seed_root>/data/task-XXXX/ep.parquet``, return
    ``<seed_root>/videos/task-XXXX`` if it exists."""
    task_dir = os.path.basename(os.path.dirname(parquet_path))
    seed_root = os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
    candidate = os.path.join(seed_root, "videos", task_dir)
    return candidate if os.path.isdir(candidate) else None


def _find_meta_path_for_parquet(parquet_path: str) -> Optional[str]:
    """Optional meta JSON for segmentation obs. Returns None if absent."""
    ep_name = os.path.splitext(os.path.basename(parquet_path))[0]
    task_dir = os.path.basename(os.path.dirname(parquet_path))
    seed_root = os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
    candidate = os.path.join(seed_root, "meta", "episodes", task_dir, f"{ep_name}.json")
    return candidate if os.path.isfile(candidate) else None


def discover_episodes(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Return a list of ``{source_parquet, source_key, task_id, source_seed}``."""
    records: List[Dict[str, str]] = []

    if args.data_dir:
        for pq in sorted(glob.glob(os.path.join(args.data_dir, "*.parquet"))):
            source_key = os.path.splitext(os.path.basename(pq))[0].split("_", 1)[1]
            records.append(
                {
                    "source_parquet": os.path.abspath(pq),
                    "source_key": source_key,
                    "task_id": _extract_task_id_from_parquet(pq),
                    "source_seed": None,
                }
            )
        return records

    suffix_part = f"-{args.suffix}" if args.suffix else ""
    for seed in args.seeds_list:
        seed_token = f"{args.seed_separator}{seed}"
        demos_name = f"{args.demos_prefix}{suffix_part}-seed{seed_token}"
        for tid in args.task_ids_list:
            task_dir = os.path.join(args.datasets_root, demos_name, "data", f"task-{tid:04d}")
            for pq in sorted(glob.glob(os.path.join(task_dir, "*.parquet"))):
                source_key = os.path.splitext(os.path.basename(pq))[0].split("_", 1)[1]
                records.append(
                    {
                        "source_parquet": os.path.abspath(pq),
                        "source_key": source_key,
                        "task_id": tid,
                        "source_seed": seed,
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Renumbering to 8-digit keys
# ---------------------------------------------------------------------------


def assign_new_keys(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Assign canonical 8-digit ``{task_id:04d}{demo_idx:04d}`` keys per task.

    The order is taken from the already-sorted ``records`` list so repeated
    runs with the same inputs give stable keys.
    """
    per_task_counter: Dict[int, int] = defaultdict(int)
    for rec in records:
        tid = rec["task_id"]
        demo_idx = per_task_counter[tid]
        per_task_counter[tid] += 1
        assert demo_idx < 10000, (
            f"More than 9999 episodes for task {tid}; 8-digit key scheme "
            f"(int(demo_key) // 10000 == task_id) would overflow."
        )
        rec["new_key"] = f"{tid:04d}{demo_idx:04d}"
    return records


# ---------------------------------------------------------------------------
# Split resolution
# ---------------------------------------------------------------------------


def _split_json_episode_candidates(ep: str) -> List[str]:
    """Normalize split_json episode ids to the forms we can resolve.

    GCBC / ``convert_to_tfrecord`` often writes ``episode_<stem>`` strings
    (e.g. ``episode_00530000001001000``). Parquet stems here use the same
    numeric part as ``source_key`` (``00530000001001000``). We try the raw
    string, then strip a leading ``episode_`` prefix if present.
    """
    s = str(ep).strip()
    prefix = "episode_"
    out: List[str] = []
    for cand in (s, s[len(prefix) :] if s.startswith(prefix) else None):
        if cand and cand not in out:
            out.append(cand)
    return out


def resolve_splits(
    records: List[Dict[str, str]], args: argparse.Namespace
) -> Dict[str, str]:
    """Return a mapping ``new_key -> split``."""
    if args.split_json:
        with open(args.split_json) as f:
            split_spec = json.load(f)
        # Build lookups from BOTH the original and the new key so the user
        # can write their split JSON using either name convention.
        original_to_new = {r["source_key"]: r["new_key"] for r in records}
        key_set = {r["new_key"] for r in records}

        key_to_split: Dict[str, str] = {}
        unknown: List[str] = []
        for split_name, ep_list in split_spec.items():
            if split_name not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split {split_name!r} in {args.split_json}")
            for ep in ep_list:
                new_key: Optional[str] = None
                for cand in _split_json_episode_candidates(ep):
                    if cand in key_set:
                        new_key = cand
                        break
                    if cand in original_to_new:
                        new_key = original_to_new[cand]
                        break
                if new_key is None:
                    unknown.append(str(ep))
                    continue
                key_to_split[new_key] = split_name

        missing = [r["new_key"] for r in records if r["new_key"] not in key_to_split]
        if missing:
            print(
                f"[warn] {len(missing)} discovered episode(s) have no matching "
                f"entry in --split_json (new_key form). They will be omitted from "
                f"the prepared output. First few new_key values: {missing[:5]}",
                file=sys.stderr,
            )
        if unknown:
            print(
                f"[warn] --split_json references {len(unknown)} episodes not "
                f"found in the discovered set (will be ignored): "
                f"first few = {unknown[:5]}"
            )
        return key_to_split

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got "
        f"{args.train_ratio + args.val_ratio + args.test_ratio}"
    )

    rng = np.random.RandomState(args.split_seed)
    n = len(records)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(n * args.test_ratio)) if args.test_ratio > 0 else 0
    n_val = max(1, int(n * args.val_ratio)) if args.val_ratio > 0 else 0
    test_idx = set(indices[:n_test].tolist())
    val_idx = set(indices[n_test : n_test + n_val].tolist())
    split_map: Dict[str, str] = {}
    for i, rec in enumerate(records):
        if i in test_idx:
            split_map[rec["new_key"]] = "test"
        elif i in val_idx:
            split_map[rec["new_key"]] = "val"
        else:
            split_map[rec["new_key"]] = "train"
    return split_map


# ---------------------------------------------------------------------------
# Copy / link helpers
# ---------------------------------------------------------------------------


def _place(src: str, dst: str, mode: str, resume: bool) -> None:
    """Symlink (default), hardlink or copy ``src`` to ``dst``.

    Respects ``resume`` by skipping if ``dst`` already exists.
    """
    if resume and (os.path.islink(dst) or os.path.exists(dst)):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown placement mode {mode!r}")


def place_episode(
    rec: Dict[str, str],
    output_dir: str,
    mode: str,
    resume: bool,
) -> Tuple[str, List[str]]:
    """Create renumbered symlinks/copies for one episode's parquet + videos.

    Returns
    -------
    new_parquet_path : str
        Absolute path of the renumbered parquet inside ``output_dir``.
    placed_video_paths : list[str]
        Absolute paths of the renumbered video files (one per modality/camera).
    """
    new_key = rec["new_key"]
    tid = rec["task_id"]
    task_dir = f"task-{tid:04d}"

    src_parquet = rec["source_parquet"]
    dst_parquet = os.path.join(
        output_dir, "2025-challenge-demos", "data", task_dir,
        f"episode_{new_key}.parquet",
    )
    _place(src_parquet, dst_parquet, mode, resume)

    placed_videos: List[str] = []
    src_video_root = _find_video_root_for_parquet(src_parquet)
    if src_video_root is not None:
        for sub in sorted(os.listdir(src_video_root)):
            sub_dir = os.path.join(src_video_root, sub)
            if not os.path.isdir(sub_dir):
                continue
            # Expect files named episode_<source_key>.mp4. Rename to new_key.
            src_name = f"episode_{rec['source_key']}.mp4"
            src_video = os.path.join(sub_dir, src_name)
            if not os.path.isfile(src_video):
                # Some modalities may not exist per-episode; just skip.
                continue
            dst_video = os.path.join(
                output_dir, "2025-challenge-demos", "videos", task_dir, sub,
                f"episode_{new_key}.mp4",
            )
            _place(src_video, dst_video, mode, resume)
            placed_videos.append(os.path.abspath(dst_video))

    # Optional meta json (for seg_instance_id).
    meta_src = _find_meta_path_for_parquet(src_parquet)
    if meta_src is not None:
        meta_dst = os.path.join(
            output_dir, "2025-challenge-demos", "meta", "episodes", task_dir,
            f"episode_{new_key}.json",
        )
        _place(meta_src, meta_dst, mode, resume)

    return os.path.abspath(dst_parquet), placed_videos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Input: either --data_dir OR --seeds + --task_ids (GCBC-parity).
    p.add_argument("--data_dir", type=str, default=None,
                   help="Single demos dir to consolidate (e.g. "
                        "datasets/ispatialgym-demos-seed-1/data/task-0051).")
    p.add_argument("--datasets_root", type=str, default="datasets",
                   help="Parent dir containing the per-seed demos/instances.")
    p.add_argument("--demos_prefix", type=str, default="ispatialgym-demos",
                   help='Demos directory prefix (before "-seed{...}").')
    p.add_argument("--instances_prefix", type=str, default="ispatialgym-instances",
                   help="(Informational only) Instances dir prefix; the goal "
                        "images are resolved via image_condition_path.")
    p.add_argument("--suffix", type=str, default=None,
                   help="Optional suffix inserted before the seed token, e.g. "
                        '"d0-goal" → ispatialgym-demos-d0-goal-seed{...}.')
    p.add_argument("--seed_separator", type=str, default="",
                   help='Token between "seed" and the seed number. Repo uses '
                        'a mix of "seed1" (default) and "seed-1".')
    p.add_argument("--seeds", type=str, default=None,
                   help="Seed range (1-20) or list (1,2,4).")
    p.add_argument("--task_ids", type=str, default=None,
                   help="Task-ID range (51-54) or list (51,52).")

    p.add_argument("--output_dir", type=str, required=True,
                   help="Consolidated output dir. Pass this as data_dir= to "
                        "train.py.")
    p.add_argument("--project_root", type=str, default=None,
                   help="Root used to resolve relative image_condition_path "
                        "values in each parquet. Only affects the generated "
                        "test_episodes.json.")

    # Placement mode.
    p.add_argument("--link_mode", choices=["symlink", "hardlink", "copy"],
                   default="symlink",
                   help="How to place files in --output_dir.")
    p.add_argument("--resume", action="store_true",
                   help="Skip episodes whose renumbered files already exist.")

    # Split args.
    p.add_argument("--split_json", type=str, default=None,
                   help='JSON {"train": [...], "val": [...], "test": [...]} '
                        "keyed by original OR new 8-digit episode IDs. "
                        "Discovered episodes not listed in any split are omitted "
                        "(logged); split entries with no matching demo are ignored.")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--include_splits", nargs="+",
                   default=["train", "val"],
                   choices=["train", "val", "test"],
                   help="Which splits to place into the consolidated demos dir. "
                        'Default ("train val") keeps held-out test episodes '
                        "out of the training dataset but still records them in "
                        "test_episodes.json for online eval.")

    args = p.parse_args()

    # Validate mutually exclusive data source options (GCBC parity).
    if args.seeds is not None or args.task_ids is not None:
        assert args.seeds and args.task_ids, \
            "--seeds and --task_ids must both be provided"
        assert args.data_dir is None, \
            "--data_dir cannot be used together with --seeds/--task_ids"
        args.seeds_list = parse_int_list(args.seeds)
        args.task_ids_list = parse_int_list(args.task_ids)
    else:
        assert args.data_dir is not None, \
            "Either --data_dir or --seeds + --task_ids is required"
        args.seeds_list = []
        args.task_ids_list = []

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Discover.
    records = discover_episodes(args)
    if not records:
        print("[error] No episodes discovered. Check --seeds / --task_ids / "
              "--data_dir and --seed_separator.", file=sys.stderr)
        return 1
    print(f"Discovered {len(records)} episodes across "
          f"{len({r['task_id'] for r in records})} tasks.")

    # 2. Renumber.
    records = assign_new_keys(records)

    # 3. Split.
    split_map = resolve_splits(records, args)
    if args.split_json:
        n_before = len(records)
        records = [r for r in records if r["new_key"] in split_map]
        n_omitted = n_before - len(records)
        if n_omitted:
            print(
                f"[info] After intersecting with --split_json: preparing "
                f"{len(records)} episode(s), omitted {n_omitted}."
            )
        if not records:
            print(
                "[error] No episodes remain after matching discovery to "
                "--split_json. Check split keys (source episode id or new_key) "
                "and that split_json lists the episodes you expect.",
                file=sys.stderr,
            )
            return 1

    # 4. Place files.
    test_meta: Dict[str, Dict[str, str]] = {}
    placed = 0
    skipped_split = 0
    test_meta_warnings: List[str] = []
    for rec in tqdm(records, desc="Placing"):
        split = split_map[rec["new_key"]]
        if split not in args.include_splits and split != "test":
            skipped_split += 1
            continue
        # Test-split episodes: still collect metadata for eval, but only
        # materialise the symlinks/copies if the user asked for them.
        if split == "test":
            meta, warnings = resolve_instance_metadata(
                rec["source_parquet"], args.project_root)
            test_meta_warnings.extend(warnings)
            if "test" in args.include_splits:
                dst_parquet, _ = place_episode(
                    rec, args.output_dir, args.link_mode, args.resume)
                placed += 1
            else:
                dst_parquet = rec["source_parquet"]
            meta.update(
                {
                    "parquet_path": os.path.abspath(dst_parquet),
                    "task_id": rec["task_id"],
                    "new_key": rec["new_key"],
                    "source_key": rec["source_key"],
                    "source_seed": rec["source_seed"],
                }
            )
            test_meta[rec["new_key"]] = meta
            continue

        place_episode(rec, args.output_dir, args.link_mode, args.resume)
        placed += 1

    # 5. Sidecar JSONs.
    split_out = {"train": [], "val": [], "test": []}
    for new_key, split in sorted(split_map.items()):
        split_out.setdefault(split, []).append(new_key)

    with open(os.path.join(args.output_dir, "episode_ids_split.json"), "w") as f:
        json.dump(split_out, f, indent=2)

    key_map = {
        rec["new_key"]: {
            "source_key": rec["source_key"],
            "source_parquet": rec["source_parquet"],
            "source_seed": rec["source_seed"],
            "task_id": rec["task_id"],
            "split": split_map[rec["new_key"]],
        }
        for rec in records
    }
    with open(os.path.join(args.output_dir, "episode_key_map.json"), "w") as f:
        json.dump(key_map, f, indent=2)

    if test_meta:
        with open(os.path.join(args.output_dir, "test_episodes.json"), "w") as f:
            json.dump(test_meta, f, indent=2)

    with open(os.path.join(args.output_dir, "prepare_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()
                   if not k.endswith("_list")}, f, indent=2, default=str)

    # Final summary.
    counts = {s: sum(1 for v in split_map.values() if v == s) for s in ["train", "val", "test"]}
    print(f"\nDone. Placed {placed} episode(s) "
          f"(skipped {skipped_split} outside --include_splits).")
    print(f"Split sizes — train: {counts['train']}, val: {counts['val']}, "
          f"test: {counts['test']}")
    print(f"Consolidated dir: {os.path.abspath(args.output_dir)}")
    if test_meta_warnings:
        print(
            f"\n[warn] {len(test_meta_warnings)} test-metadata warning(s); "
            "test_episodes.json is still written with whatever fields could "
            "be resolved. First few:"
        )
        for w in test_meta_warnings[:5]:
            print(f"  - {w}")
    print(
        "\nTrain with:\n"
        f"  python train.py --config-name base_config_goal_image \\\n"
        f"      arch=act_goal task=<task_name> robot=r1pro \\\n"
        f"      data_dir={os.path.abspath(args.output_dir)} \\\n"
        f"      goal_image_project_root="
        f"{os.path.abspath(args.project_root) if args.project_root else '<abs_repo_root>'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

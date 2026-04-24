# il_library
Imitation Learning Library with Pytorch Lightning.

## ACT + goal-image baseline

`il_lib` ships a goal-conditioned ACT variant that adds a per-episode goal
reference image as an additional multi-view input, without modifying the ACT
policy or `MultiviewResNet18` internals.

### Data flow

```
head camera RGB (current frame)            goal image (episode reference, PNG)
       │                                               │
       ├─ proprioception (qpos / eef / odom) ──────────┤
       │                                               │
       ▼                                               ▼
┌──────────────────── MultiviewResNet18 ────────────────────┐
│   view = head                                             │
│   view = goal  ◀── goal image injected as an extra view   │
└───────────────────────────────────────────────────────────┘
                           │
                           ▼
               ACT transformer (CVAE, chunked action head)
```

The wrist cameras from `robot/r1pro.yaml` are intentionally dropped so the
visual inputs exactly mirror the GCBC goal-image pipeline (head RGB + goal
RGB).

The goal image is **not** channel-stacked onto the current observation. It is
treated as a first-class multi-view input: the shared ResNet18 backbone
embeds each view independently and ACT concatenates the per-view feature
maps along the width axis before the transformer.

### Components

| File | Role |
| --- | --- |
| `il_lib/datas/iterable_dataset_goal_image.py` | `BehaviorIterableDatasetWithGoalImage` — loads the goal PNG from the parquet column `image_condition_path` and injects it as `{goal_view_name}::rgb` with shape `(L, 3, H, W)`. |
| `il_lib/policies/policy_base_goal_image.py` | `PolicyWrapperWithGoalImage` — caches a per-episode goal image (constructor kwarg, `ACT_GOAL_IMAGE_PATH` env var, `set_goal_image(path)`, or `policy.goal_image = <np.ndarray / path>`) and appends it to every processed observation for online eval. The `goal_image` property is compatible with the existing OmniGibson eval harness, which assigns `policy.goal_image = <uint8 HWC np.ndarray>` per episode. |
| `il_lib/configs/arch/act_goal.yaml` | ACT arch config: RGB-only (`include_depth: false`), `obs_backbone.views = [head, goal]`. |
| `il_lib/configs/base_config_goal_image.yaml` | Top-level Hydra config: swaps `data.dataset_class` and `policy_wrapper._target_` to the goal-image variants, overrides `multi_view_cameras` to head-only, adds `goal_view_name` / `goal_image_project_root`. |
| `prepare_data.py` | Consolidates scattered `ispatialgym-demos{-suffix}-seed{N}/` directories into the single `2025-challenge-demos/` layout that `BehaviorIterableDataset` expects, renumbering episodes to the required 8-digit keys. Supports `--split_json` for custom train/val/test splits. |

ACT itself (`il_lib.policies.ACT`) is unchanged; `process_data` already
routes every obs key containing `"rgb"` through the multi-view backbone, so
adding the goal view name to `obs_backbone.views` and injecting the obs key
is enough.

### Data layout expected by `train.py`

`BehaviorIterableDataset` requires this structure under `data_dir`:

```
<data_dir>/2025-challenge-demos/
    data/task-XXXX/episode_NNNNNNNN.parquet
    videos/task-XXXX/observation.images.<modality>.<camera_id>/episode_NNNNNNNN.mp4
    meta/episodes/task-XXXX/episode_NNNNNNNN.json    # only needed for seg_instance_id
```

Episode keys must be **8 digits**: `{task_id:04d}{demo_idx:04d}`. The dataset
derives `task_id = int(demo_key) // 10000`, so longer keys (like the 14/16-
digit names the iSpatialGym generator writes) will break this math and need
renumbering. See `prepare_data.py`.

### Prepare the dataset

The iSpatialGym demo generator writes per-seed directories:

```
datasets/
    ispatialgym-demos{suffix}-seed{N}/     # parquet + videos
        data/task-XXXX/episode_<long>.parquet
        videos/task-XXXX/observation.images.<mod>.<cam>/episode_<long>.mp4
    ispatialgym-instances{suffix}-seed{N}/ # BDDL, templates, TRO state, goal PNGs
        task-XXXX/<instance>/
            bddl/*.bddl
            *_template.json
            *-tro_state.json
            reference_image2.png
```

`prepare_data.py` walks these scattered dirs, filters by seed / task_id /
split, renumbers each episode to an 8-digit key, and symlinks (or copies)
everything into a single output directory that `train.py` can consume
directly.

```bash
python prepare_data.py \
    --datasets_root /abs/path/to/behavior-1k-private/datasets \
    --demos_prefix ispatialgym-demos \
    --seed_separator - \
    --seeds 1-10 \
    --task_ids 51-54 \
    --output_dir /abs/path/to/datasets/act-goal-v1 \
    --project_root /abs/path/to/behavior-1k-private \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

Flags that match `convert_to_tfrecord.py` 1:1: `--data_dir`, `--seeds`,
`--task_ids`, `--suffix`, `--train_ratio` / `--val_ratio` / `--test_ratio`,
`--split_seed`, `--split_json`, `--resume`. Extras specific to this script:

- `--seed_separator` — handles both `ispatialgym-demos-seed-1` (default) and
  `ispatialgym-demos-seed1` (set to `""`).
- `--link_mode {symlink,hardlink,copy}` — symlinks by default (zero disk
  duplication); use `copy` if the sources will be moved/deleted.
- `--include_splits train val` (default) — test episodes are still listed
  in `test_episodes.json` for the eval harness but kept out of the training
  set. Pass `train val test` to materialise everything.

Sidecar JSONs written under `--output_dir`:

- `episode_ids_split.json` — `{"train": [...], "val": [...], "test": [...]}`
  keyed by the **new** 8-digit IDs. Format matches GCBC's
  `convert_to_tfrecord.py` output so the same file can seed either pipeline.
- `episode_key_map.json` — maps new 8-digit keys back to
  `{source_parquet, source_seed, source_key, task_id, split}` for audit.
- `test_episodes.json` — per held-out episode: `bddl_path`, `template_path`,
  `tro_path`, `goal_image_path`, and the renumbered parquet path; this is
  the input format `eval_ispatialgym{,_batched}.py` consumes.
- `prepare_config.json` — the exact CLI args, for reproducibility.

`--split_json` accepts the GCBC split format (same
`{"train":[...], "val":[...], "test":[...]}` schema). Episode IDs inside it
can be either the **original** long names or the **new** 8-digit keys; the
script recognises both.

### Train

```bash
python train.py \
    --config-name base_config_goal_image \
    arch=act_goal \
    task=<task_name> \
    robot=r1pro \
    data_dir=/abs/path/to/act-goal-v1 \
    goal_image_project_root=/abs/path/to/behavior-1k-private
```

`<task_name>` is the **string name** of the task, not the numeric ID. The
allowed values are the keys of `TASK_NAMES_TO_INDICES` in
`OmniGibson/omnigibson/learning/utils/eval_utils.py` (`task_id = 0..54`):
`turning_on_radio`, `picking_up_trash`, …, `camera_relocalization` (51),
`path_integration` (52), `object_scaling` (53), `mental_rotation` (54).
Hydra resolves this against `il_lib/configs/task/<task_name>.yaml`.

- `data_dir` points at the directory produced by `prepare_data.py` (the one
  that contains `2025-challenge-demos/...`).
- `goal_image_project_root` is used to resolve relative
  `image_condition_path` entries in each parquet to an absolute PNG on disk.
  If the parquet already stores absolute paths it is ignored.

**Fair comparison vs GCBC:** `base_config_goal_image.yaml` matches GCBC on
*protocol*—same gradient budget (`max_steps: 100000`), same log / validation /
checkpoint intervals (1000 / 2000 / 2000 steps), same no-augment policy on
the ResNet views, same `wandb_project` (`ispatialgym-gcbc-torch`) for
comparable curves, and the same data layout / splits as GCBC when using
`prepare_data.py` + `generate_clone_split.py`. **ACT baseline fidelity:** learning
rate (`7e-4`), batch size (`128`), weight decay (`0.1` in `arch/act_goal.yaml`),
cosine schedule length, chunk horizon, transformer depth, KL weight, and
temporal ensemble follow `il_lib`’s ACT defaults (not GCBC’s `3e-4` / 256). For an
optional “same optimizer hyperparams as GCBC” ablation, run with e.g.
`lr=3e-4 bs=256` on the CLI. Slurm example: `scripts/train_act_goal.sh`.

Smoke test:

```bash
python train.py \
    --config-name base_config_goal_image \
    arch=act_goal task=camera_relocalization robot=r1pro \
    data_dir=/abs/path/to/act-goal-v1 \
    goal_image_project_root=/abs/path/to/behavior-1k-private \
    trainer.fast_dev_run=true bs=2 use_wandb=false
```

### Online eval — set the goal image per episode

The wrapper exposes a `goal_image` property that matches the attribute the
existing OmniGibson eval harness already uses for goal-conditioned policies
(see `eval_ispatialgym.py` / `eval_ispatialgym_batched.py`), so **no changes
to the eval harness are required**. Any one of these works:

```bash
# via env var (set once per eval job)
export ACT_GOAL_IMAGE_PATH=/abs/path/to/episode_goal.png

# via Hydra CLI override at construction time
python train.py --config-name base_config_goal_image ... \
    policy_wrapper.goal_image_path=/abs/.../goal.png
```

```python
# programmatically, per episode — this is what the OmniGibson harness does:
#   goal_img = np.array(Image.open(path).convert("RGB"))
#   policy.goal_image = goal_img
# Both of these also work:
policy_wrapper.set_goal_image("/abs/path/to/goal.png")
policy_wrapper.goal_image = "/abs/path/to/goal.png"          # path
policy_wrapper.goal_image = uint8_hwc_numpy_array            # preloaded image
```

The setter accepts a filesystem path, a uint8 `HxWx3` numpy array, or a
torch tensor; it resizes to the head-camera resolution internally so
callers can pass any source resolution.

### Notes / assumptions

- **RGB-only, head + goal.** Two visual views total. Matches the GCBC
  pipeline (`il/bridge_data_v2`), which also uses a single obs RGB + goal
  RGB with no depth. If you want to reintroduce wrist cameras, add them
  back to both `module.obs_backbone.views` (in `arch/act_goal.yaml`) and
  `head_only_cameras` (in `base_config_goal_image.yaml`).
- **Shared spatial dims.** All views are resized to the head camera
  resolution (default `240×240`). `MultiviewResNet18` and the positional
  encoding assume this.
- **Parquet column.** The dataset reads `image_condition_path` — the same
  column the GCBC pipeline in `il/bridge_data_v2` produces. Override
  `goal_image_column` via the data module kwargs if needed.
- **One goal per demo.** The first row of the column is used.
- **Normalization.** ACT keeps `il_lib`'s in-model min–max normalization
  (`/255` for RGB, `(x - low) / (high - low)` to `[-1, 1]` for proprio and
  actions using the constants in
  `OmniGibson/omnigibson/learning/utils/eval_utils.py`). No per-dataset
  `action_proprio_metadata.json` is loaded; GCBC's z-score scheme is
  intentionally *not* ported in this baseline.

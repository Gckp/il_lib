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

ACT itself (`il_lib.policies.ACT`) is unchanged; `process_data` already
routes every obs key containing `"rgb"` through the multi-view backbone, so
adding the goal view name to `obs_backbone.views` and injecting the obs key
is enough.

### Train

```bash
python train.py \
    --config-name base_config_goal_image \
    arch=act_goal \
    task=<task_name> \
    robot=r1pro \
    data_dir=/abs/path/to/behavior_demos \
    goal_image_project_root=/abs/path/to/behavior_demos
```

- `data_dir` points at the directory containing
  `2025-challenge-demos/data/task-XXXX/episode_YYYY.parquet` (the standard
  OmniGibson demo layout).
- `goal_image_project_root` is used to resolve relative
  `image_condition_path` entries in each parquet to an absolute PNG on disk.
  If the parquet already stores absolute paths it is ignored.

Smoke test:

```bash
python train.py \
    --config-name base_config_goal_image \
    arch=act_goal task=<task_name> robot=r1pro \
    data_dir=/abs/path/to/behavior_demos \
    goal_image_project_root=/abs/path/to/behavior_demos \
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

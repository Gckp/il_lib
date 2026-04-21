"""Goal-image aware BehaviorIterableDataset.

Extends OmniGibson's ``BehaviorIterableDataset`` so that each sample has, in
addition to the usual multi-view observation frames, a per-episode goal
reference image injected as a new named view.

Design:
    - Each parquet demo file has an ``image_condition_path`` column whose first
      entry points to a goal reference PNG. This is the same convention the
      goal-conditioned GCBC / diffusion baselines in ``il/bridge_data_v2`` use.
    - For ACT, we do NOT channel-stack obs + goal. Instead we follow the
      ``MultiviewResNet18`` convention: the goal image is loaded once per demo,
      tensorised to uint8 ``(3, H, W)`` and broadcast across the obs window to
      produce a new ``(L, 3, H, W)`` view with a dedicated key.
    - ``il_lib``'s ACT policy routes all obs keys containing the substring
      ``rgb`` through the multi-view backbone via ``process_data``, so
      registering the goal view with a key like ``{goal_view_name}::rgb`` and
      listing the same view name in ``obs_backbone.views`` is sufficient.

The goal view name / camera key are configurable so the same class can be used
with any robot / camera naming convention.
"""

from __future__ import annotations

import os
from typing import Any, Generator, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from PIL import Image

from omnigibson.learning.datas.iterable_dataset import BehaviorIterableDataset


__all__ = ["BehaviorIterableDatasetWithGoalImage"]


# Default goal view name. Chosen to be consistent with existing R1Pro camera
# view names (see OmniGibson/omnigibson/learning/configs/robot/r1pro.yaml),
# which use ``robot_r1::<link>:Camera:0``. The suffix ``::rgb`` is appended in
# the obs dict key so ACT.process_data routes it through the multi-view
# backbone automatically.
DEFAULT_GOAL_VIEW_NAME = "robot_r1::goal::Camera:0"


class BehaviorIterableDatasetWithGoalImage(BehaviorIterableDataset):
    """``BehaviorIterableDataset`` with a per-demo goal reference image view.

    Args:
        goal_view_name: The view name (no modality suffix) used to register
            the goal image. The obs dict key for the goal RGB will be
            ``f"{goal_view_name}::rgb"``. Include the same string in the ACT
            backbone's ``views:`` list so ``MultiviewResNet18`` accepts it.
        goal_image_size: ``(H, W)`` used to resize the goal image. Defaults
            to the head camera's ``resolution`` from ``multi_view_cameras``
            (falling back to ``(240, 240)``) so all views share spatial dims,
            which the shared ResNet18 backbone and the spatial position
            encoding assume.
        project_root: Optional prefix for resolving relative
            ``image_condition_path`` values stored in the parquet. If the
            parquet path is absolute this is ignored.
        goal_image_column: Parquet column holding the goal image path. The
            default matches the convention used by the GCBC pipeline.
    """

    def __init__(
        self,
        *args,
        goal_view_name: str = DEFAULT_GOAL_VIEW_NAME,
        goal_image_size: Optional[Tuple[int, int]] = None,
        project_root: Optional[str] = None,
        goal_image_column: str = "image_condition_path",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._goal_view_name = goal_view_name
        self._goal_obs_key = f"{goal_view_name}::rgb"
        self._goal_image_column = goal_image_column
        self._project_root = project_root

        # Resolve goal image resolution. Default to the head camera res so all
        # views share H/W, which is required by the shared MultiviewResNet18.
        if goal_image_size is None:
            resolved = None
            if self._multi_view_cameras is not None:
                if "head" in self._multi_view_cameras:
                    resolved = tuple(self._multi_view_cameras["head"]["resolution"])
                else:
                    first_cam = next(iter(self._multi_view_cameras.values()))
                    resolved = tuple(first_cam["resolution"])
            self._goal_image_size = resolved if resolved is not None else (240, 240)
        else:
            self._goal_image_size = tuple(goal_image_size)

        # Preload a tensor per demo. Each entry is uint8 (3, H, W).
        self._goal_images = [
            self._load_goal_image(demo_key) for demo_key in self._demo_keys
        ]

    def _load_goal_image(self, demo_key: Any) -> th.Tensor:
        """Load and cache a single goal reference image tensor (3, H, W)."""
        task_id = int(demo_key) // 10000
        parquet_path = os.path.join(
            self._data_path,
            "2025-challenge-demos",
            "data",
            f"task-{task_id:04d}",
            f"episode_{demo_key}.parquet",
        )
        df = pd.read_parquet(parquet_path, columns=[self._goal_image_column])
        goal_path = df[self._goal_image_column].iloc[0]

        if isinstance(goal_path, bytes):
            goal_path = goal_path.decode("utf-8")
        if not os.path.isabs(goal_path) and self._project_root is not None:
            goal_path = os.path.join(self._project_root, goal_path)

        img = Image.open(goal_path).convert("RGB")
        H, W = self._goal_image_size
        # PIL ``resize`` takes (W, H)
        img = img.resize((W, H))
        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        tensor = th.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W) uint8
        return tensor

    def get_streamed_data(
        self, demo_ptr: int, start_idx: int, end_idx: int
    ) -> Generator[dict, None, None]:
        """Wrap the parent generator to attach the goal image view per sample."""
        goal_image = self._goal_images[demo_ptr]  # (3, H, W) uint8
        # (obs_window_size, 3, H, W) view onto the cached tensor.
        goal_window = goal_image.unsqueeze(0).expand(
            self._obs_window_size, -1, -1, -1
        )
        for data in super().get_streamed_data(demo_ptr, start_idx, end_idx):
            # ACT.process_data picks up any obs key containing "rgb" and
            # keys the resulting multi-view dict by ``k.rsplit("::", 1)[0]``.
            # Registering the key here with the matching view name makes the
            # goal image an additional MultiviewResNet18 input alongside the
            # wrist / head cameras with no changes needed in ACT itself.
            data["obs"][self._goal_obs_key] = goal_window
            yield data

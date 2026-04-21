"""Goal-image aware ``PolicyWrapper`` for online evaluation.

At training time the dataset
(``il_lib.datas.BehaviorIterableDatasetWithGoalImage``) injects the goal image
into the observation dict. At rollout time the environment only emits per-step
observations, so this wrapper is responsible for loading a per-episode goal
image and appending it to every processed observation under the same key.

The wrapper is compatible with the existing OmniGibson eval harness
(``eval_ispatialgym.py`` / ``eval_ispatialgym_batched.py``), which updates
per-episode goals by assigning directly to ``policy.goal_image``. It also
supports the two legacy entry points (constructor ``goal_image_path`` and the
``ACT_GOAL_IMAGE_PATH`` env var) for stand-alone scripts.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from PIL import Image

from il_lib.policies.policy_base import PolicyWrapper


__all__ = ["PolicyWrapperWithGoalImage"]


DEFAULT_GOAL_VIEW_NAME = "robot_r1::goal::Camera:0"


GoalImageLike = Union[str, os.PathLike, np.ndarray, th.Tensor]


class PolicyWrapperWithGoalImage(PolicyWrapper):
    """Extend ``PolicyWrapper`` with a per-episode goal reference image.

    The goal image is cached once per episode and added to every processed
    observation under ``f"{goal_view_name}::rgb"`` with the same
    ``(1, 1, 3, H, W)`` shape the parent wrapper produces for each camera
    view. ``il_lib.policies.ACT.process_data`` then routes it through the
    multi-view backbone just like any other camera.

    Supported ways to set the goal (in order of precedence at construction):
      1. ``ACT_GOAL_IMAGE_PATH`` environment variable (path to PNG/JPEG).
      2. ``goal_image_path`` constructor kwarg (same).
      3. Assigning to the ``goal_image`` property after construction. This is
         the path the OmniGibson eval harness uses; it checks
         ``hasattr(policy, "goal_image")`` and assigns a uint8 HWC numpy
         array per episode.
      4. Calling :meth:`set_goal_image` with a filesystem path.
    """

    def __init__(
        self,
        *args,
        goal_view_name: str = DEFAULT_GOAL_VIEW_NAME,
        goal_image_path: Optional[str] = None,
        goal_image_size: Optional[Tuple[int, int]] = None,
        goal_image_env_var: str = "ACT_GOAL_IMAGE_PATH",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._goal_view_name = goal_view_name
        self._goal_obs_key = f"{goal_view_name}::rgb"

        # Fall back to the head camera resolution so all MultiviewResNet18
        # inputs share spatial dims.
        if goal_image_size is None:
            if "head" in self.obs_output_size:
                goal_image_size = self.obs_output_size["head"]
            else:
                goal_image_size = next(iter(self.obs_output_size.values()))
        self._goal_image_size: Tuple[int, int] = tuple(goal_image_size)

        # Cached raw uint8 HWC array, returned by the ``goal_image`` getter.
        # Kept in sync with ``self._goal_tensor`` (the preprocessed (1,1,3,H,W)
        # tensor that actually gets injected into processed observations).
        self._goal_image_np: Optional[np.ndarray] = None
        self._goal_tensor: Optional[th.Tensor] = None

        env_override = os.environ.get(goal_image_env_var)
        resolved_path = env_override or goal_image_path
        if resolved_path is not None:
            self.set_goal_image(resolved_path)

    # ------------------------------------------------------------------
    # Public API used by the eval harness
    # ------------------------------------------------------------------

    @property
    def image_size(self) -> Optional[int]:
        """Square image side length, if the goal view is square.

        The OmniGibson eval harness checks ``hasattr(policy, "image_size")``
        and, if present, pre-resizes the goal image to ``(image_size,
        image_size)`` before assigning to ``policy.goal_image``. Exposing it
        keeps that fast-path working; for non-square goal sizes we return
        ``None`` so the harness skips the pre-resize and our setter handles
        it instead.
        """
        H, W = self._goal_image_size
        return int(H) if H == W else None

    @property
    def goal_image(self) -> Optional[np.ndarray]:
        """Return the cached goal image as a uint8 HWC numpy array.

        Returns ``None`` until a goal image has been set.
        """
        return self._goal_image_np

    @goal_image.setter
    def goal_image(self, value: Optional[GoalImageLike]) -> None:
        """Set the goal image from a path, numpy array, or tensor.

        This is the entry point used by the OmniGibson eval harness, which
        assigns a uint8 HWC numpy array per episode.
        """
        if value is None:
            self._goal_image_np = None
            self._goal_tensor = None
            return
        if isinstance(value, (str, os.PathLike)):
            self.set_goal_image(str(value))
            return
        if th.is_tensor(value):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        self._set_goal_from_array(arr)

    def set_goal_image(self, goal_image_path: str) -> None:
        """Load (or reload) the goal image from disk."""
        img = Image.open(goal_image_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        self._set_goal_from_array(arr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_goal_from_array(self, arr: np.ndarray) -> None:
        """Resize ``arr`` (HWC uint8 or similar) and cache as (1,1,3,H,W)."""
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"Goal image must be HxWx3, got shape {tuple(arr.shape)}."
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        H, W = self._goal_image_size
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = np.array(Image.fromarray(arr).resize((W, H)))

        self._goal_image_np = arr

        # Final shape ``(1, 1, 3, H, W)``: (B=1, obs_window=1, C, H, W). The
        # extra leading dims match the per-camera RGB tensors produced by
        # ``PolicyWrapper.process_obs`` before ``any_concat`` in ``act``.
        tensor = th.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        self._goal_tensor = self._post_processing_fn(tensor.to(th.float32))

    # ------------------------------------------------------------------
    # PolicyWrapper hooks
    # ------------------------------------------------------------------

    def process_obs(self, obs: dict) -> dict:
        processed_obs = super().process_obs(obs)
        if self._goal_tensor is None:
            raise RuntimeError(
                "Goal image has not been set. Assign to policy.goal_image, "
                "call policy_wrapper.set_goal_image(path), or set the "
                "ACT_GOAL_IMAGE_PATH env var before the first act() call."
            )
        processed_obs[self._goal_obs_key] = self._goal_tensor
        return processed_obs

    def reset(self) -> None:
        super().reset()
        # Keep the cached goal image across resets; the eval harness
        # overwrites ``goal_image`` per episode when the goal changes.

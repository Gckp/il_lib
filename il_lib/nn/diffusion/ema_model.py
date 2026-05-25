"""Exponential Moving Average helper for Diffusion Policy training.

Ports the EMA mechanics from the canonical Diffusion Policy reference
(Chi et al. 2023, ``diffusion_policy-main/diffusion_policy/model/diffusion/
ema_model.py``). The original DP implementation stores a ``deepcopy`` of the
training model inside the helper and updates that copy in place. Here we
decouple ownership so callers can keep the EMA copies as first-class
submodule attributes on a ``LightningModule`` -- doing so makes the EMA
weights flow through Lightning's state-dict / checkpoint pipeline for free,
without writing custom ``on_save_checkpoint`` hooks.

Usage pattern (called from ``il_lib.policies.DiffusionPolicy``):

    self.ema_feature_extractor = copy.deepcopy(self.feature_extractor)
    self.ema_backbone = copy.deepcopy(self.backbone)
    for m in (self.ema_feature_extractor, self.ema_backbone):
        m.eval(); m.requires_grad_(False)
    self._ema_helper = EMAModel(power=0.75, ...)

    # After each optimizer step:
    self._ema_helper.step(
        live_modules=[self.feature_extractor, self.backbone],
        ema_modules=[self.ema_feature_extractor, self.ema_backbone],
        optimization_step=int(self.ema_step.item()),
    )

The decay schedule matches the canonical implementation; defaults are taken
from ``train_diffusion_unet_image_workspace.yaml``
(``power=0.75``, ``inv_gamma=1.0``, ``max_value=0.9999``).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


__all__ = ["EMAModel"]


class EMAModel:
    """Stateless EMA scheduler. State (averaged weights, step counter) lives
    on the caller (the ``LightningModule``) so it flows through Lightning
    checkpointing without extra hooks.
    """

    def __init__(
        self,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        """EMA warmup schedule. Defaults match DP UNet image workspace.

        Args:
            update_after_step: Suppress EMA updates for the first N steps
                (warmup proper happens via the decay schedule below; this is
                an additional explicit suppression).
            inv_gamma: Inverse gamma for the EMA warmup. Default 1.0.
            power: Exponent in the warmup decay schedule. ``power=0.75`` (DP
                default for image UNet) reaches decay=0.999 at ~10k steps and
                decay=0.9999 at ~215k steps; appropriate for our ``max_steps
                =100000``. Use ``power=2/3`` for training >>1M steps.
            min_value: Decay floor (set to 0 to allow early steps to track
                live weights nearly verbatim).
            max_value: Decay ceiling (e.g. 0.9999 == EMA blends in 0.01% of
                each live step once warmed up).
        """
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        # Last decay value used; updated in ``step`` for diagnostics.
        self.last_decay = 0.0

    def get_decay(self, optimization_step: int) -> float:
        """Compute the EMA decay factor for the current optimizer step.

        Mirrors the canonical DP schedule:
            decay(step) = 1 - (1 + step / inv_gamma) ** -power
        clamped to ``[min_value, max_value]`` and forced to 0 for the very
        first step so the EMA copy starts identical to the live weights.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(
        self,
        live_modules: Sequence[nn.Module],
        ema_modules: Sequence[nn.Module],
        optimization_step: int,
    ) -> float:
        """Update each ``ema_modules[i]`` from ``live_modules[i]`` in lockstep.

        Both sequences must be the same length and each pair must be a
        ``deepcopy`` of the other (so ``.modules()`` traversal order is
        identical). Returns the decay factor applied this step (for logging).

        Iteration follows the canonical DP implementation exactly:
          * Walk modules with ``.modules()`` so we can branch on module type
            (specifically ``_BatchNorm``).
          * For each non-recursive parameter, blend toward live with
            ``ema = decay * ema + (1 - decay) * live`` unless the module is
            BatchNorm or the parameter has ``requires_grad=False``, in which
            case we copy verbatim. (BatchNorm running stats are buffers, not
            params -- they are not touched here; the BN affine ``weight`` /
            ``bias`` *are* params and get the copy treatment. The DP authors'
            recommendation is to use GroupNorm anyway, which sidesteps this
            entirely.)
        """
        assert len(live_modules) == len(ema_modules), (
            f"live/ema module lists differ in length: "
            f"{len(live_modules)} vs {len(ema_modules)}"
        )
        decay = self.get_decay(optimization_step)
        self.last_decay = decay
        for live_root, ema_root in zip(live_modules, ema_modules):
            for live_m, ema_m in zip(live_root.modules(), ema_root.modules()):
                for live_p, ema_p in zip(
                    live_m.parameters(recurse=False),
                    ema_m.parameters(recurse=False),
                ):
                    if isinstance(live_m, nn.modules.batchnorm._BatchNorm):
                        ema_p.copy_(live_p.detach().to(dtype=ema_p.dtype).data)
                    elif not live_p.requires_grad:
                        ema_p.copy_(live_p.detach().to(dtype=ema_p.dtype).data)
                    else:
                        ema_p.mul_(decay)
                        ema_p.add_(
                            live_p.detach().to(dtype=ema_p.dtype).data,
                            alpha=1.0 - decay,
                        )
        return decay

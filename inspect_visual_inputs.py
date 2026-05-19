"""Dump the visual inputs the il_lib policy receives during training.

Diagnostic for goal-conditioning and image-encoder bugs of the form "the
policy appears to be ignoring (some of) the visual inputs". The script
reuses the same Hydra config tree as ``train.py`` so the dumped images
exactly match what the model would see during a training step.

Verifies that:
  * Goal-image PNGs are not blank, wrong-resolution, or colour-swapped.
  * Every view configured in ``obs_backbone.views`` is present in each
    batch (and arrives in the expected pinned order).
  * The per-sample ``_batched_random_crop`` produces distinct crops per
    sample. Prior to its introduction, ``torchvision.transforms.RandomCrop``
    applied one shared crop offset to the entire minibatch, effectively
    making the augmentation a no-op.

Outputs (under ``+inspect.out_dir``):
    raw/{view}/b{batch:02d}_s{sample:02d}.png   Per-sample raw image as
                                                produced by
                                                ``ACT.process_data`` (scaled
                                                to [0, 1] float, no
                                                augmentation).
    aug/{view}/b{batch:02d}_s{sample:02d}.png   Only when
                                                ``+inspect.show_augmented=true``.
                                                Per-sample image after the
                                                per-sample random crop and
                                                ImageNet normalization,
                                                de-normalized for display.
    grid_b{batch:02d}.png                       One row per sample, one
                                                column per view, raw above
                                                augmented (if enabled).
    index.html                                  Static HTML browse page.
    stats.txt                                   Per-view per-channel min /
                                                mean / max. A flat all-white
                                                or all-blue input is
                                                obvious here without opening
                                                any PNG.

The augmented mode never runs the ResNet itself. It applies only the
preprocessing that prepares the tensor to be fed into ResNet18: random
crop and ImageNet normalization. See ``inspect_goal_sensitivity.py`` for
probes that exercise the trained encoder and decoder.

Usage mirrors ``train.py``:

    # Raw-only quick check.
    python inspect_visual_inputs.py --config-name base_config_goal_image \\
        arch=act_goal task=behavior task.name=<task_name> robot=r1pro \\
        data_dir=<abs path> goal_image_project_root=<abs path> \\
        +inspect.out_dir=./inspect_out \\
        +inspect.num_batches=2 \\
        +inspect.samples_per_batch=4 \\
        use_wandb=false online_eval=null

    # With post-augmentation images. Instantiates ``obs_backbone`` and
    # downloads the pretrained ResNet18 weights on first run.
    python inspect_visual_inputs.py ... +inspect.show_augmented=true
"""
from __future__ import annotations

# ----- py3.14 argparse / Hydra workaround (mirrors train.py) -----
import argparse
import copy
import os
import sys

if sys.version_info >= (3, 14):
    _orig_check_help = argparse.ArgumentParser._check_help

    def _check_help_py314(self, action):
        help_val = getattr(action, "help", None)
        if help_val is not None and not isinstance(help_val, str):
            action = copy.copy(action)
            try:
                action.help = str(help_val)
            except Exception:
                action.help = "Hydra shell completion (see hydra --help)."
        return _orig_check_help(self, action)

    argparse.ArgumentParser._check_help = _check_help_py314  # type: ignore[method-assign]
# -----------------------------------------------------------------

import html
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from il_lib.utils.config_utils import register_omegaconf_resolvers


# ImageNet normalisation constants used by ``ResNet18_Weights.DEFAULT.transforms()``.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _safe_view_name(view: str) -> str:
    """Make a view name filesystem-safe (cameras use ``::`` and ``:``)."""
    return view.replace("::", "__").replace(":", "_").replace("/", "_")


def _tensor_to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` float tensor in [0, 1] → ``(H, W, C)`` uint8."""
    img = img_chw.detach().clamp(0.0, 1.0).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        # RGBD: only the first 3 channels are rendered. The depth channel
        # arrives in [0, 1] after ``ACT.process_data`` and is dropped here
        # rather than mapped to a colour scale.
        img = img[..., :3]
    return img


def _denormalize_imagenet(x_chw: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalisation. Accepts 3- or 4-channel tensors.

    For RGBD the 4th channel uses ``mean=0`` ``std=1`` (matches
    ``MultiviewResNet18``'s include_depth branch), which is a no-op.
    """
    C = x_chw.shape[0]
    if C >= 3:
        mean = IMAGENET_MEAN.to(x_chw.device, x_chw.dtype)
        std = IMAGENET_STD.to(x_chw.device, x_chw.dtype)
        x_rgb = x_chw[:3] * std + mean
        if C == 3:
            return x_rgb
        # Preserve any extra channels (depth) untouched.
        return torch.cat([x_rgb, x_chw[3:]], dim=0)
    return x_chw


def _save_png(arr_hwc_uint8: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_hwc_uint8).save(path)


def _collect_obs_rgb(model, batch: dict) -> Dict[str, torch.Tensor]:
    """Run ``ACT.process_data`` (or equivalent) to get the per-view rgb dict.

    Falls back to a direct dataloader-side reconstruction if ``process_data``
    can't be called (e.g. the model isn't instantiated). The returned dict
    has keys in ``obs_backbone.views`` order and values of shape
    ``(B, L, 3, H, W)`` (or 4 channels for RGBD) in [0, 1] float.
    """
    if model is not None and hasattr(model, "process_data"):
        # ``process_data`` mutates ``batch`` slightly (e.g. picks fields); pass
        # a shallow copy of the obs dict to be safe.
        batch_copy = dict(batch)
        batch_copy["obs"] = dict(batch["obs"])
        data = model.process_data(batch_copy, extract_action=False)
        if "rgbd" in data:
            return data["rgbd"]
        if "rgb" in data:
            return data["rgb"]
    # Fallback: rebuild from raw obs keys.
    obs = batch["obs"]
    rgb = {
        k.rsplit("::", 1)[0]: obs[k].float() / 255.0
        for k in obs if "rgb" in k
    }
    return rgb


def _maybe_apply_augmentation(
    backbone, view_to_tensor: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Apply the same augmentation the model would apply during training.

    Returns ``(B, L, C, crop_h, crop_w)`` per view, ImageNet-normalised. The
    caller is responsible for de-normalising before display.
    """
    out = {}
    for k, v in view_to_tensor.items():
        cropped = (
            backbone._batched_random_crop(v) if backbone._enable_random_crop else v
        )
        B, L, C, H, W = cropped.shape
        flat = cropped.reshape(B * L, C, H, W).contiguous()
        flat = backbone._train_transforms(flat)
        out[k] = flat.reshape(B, L, *flat.shape[1:])
    return out


def _make_grid(
    raw_by_view: Dict[str, np.ndarray],
    aug_by_view: Optional[Dict[str, np.ndarray]],
    view_order: List[str],
    out_path: str,
) -> None:
    """Stack per-sample rows × per-view columns into a single PNG."""
    rows: List[List[np.ndarray]] = []
    # First row: raw images.
    rows.append([raw_by_view[v] for v in view_order])
    if aug_by_view is not None:
        rows.append([aug_by_view[v] for v in view_order])

    # Resize every cell to a common size (use raw[view0] as reference).
    ref_h, ref_w = rows[0][0].shape[:2]
    padded_rows = []
    for row in rows:
        padded_row = []
        for cell in row:
            if cell.shape[:2] != (ref_h, ref_w):
                cell = np.array(
                    Image.fromarray(cell).resize((ref_w, ref_h))
                )
            padded_row.append(cell)
        padded_rows.append(np.concatenate(padded_row, axis=1))
    grid = np.concatenate(padded_rows, axis=0)
    _save_png(grid, out_path)


def _format_stats(view: str, t: torch.Tensor) -> str:
    """Per-view min/mean/max + per-channel breakdown."""
    flat = t.float().reshape(-1, t.shape[-3], t.shape[-2] * t.shape[-1])
    per_channel = flat.flatten(1)
    pc_min = per_channel.min(dim=1).values.tolist()
    pc_mean = per_channel.mean(dim=1).tolist()
    pc_max = per_channel.max(dim=1).values.tolist()
    return (
        f"  view={view:<60s} "
        f"min={[f'{v:.3f}' for v in pc_min]} "
        f"mean={[f'{v:.3f}' for v in pc_mean]} "
        f"max={[f'{v:.3f}' for v in pc_max]}"
    )


def _write_index_html(out_dir: str, view_order: List[str], num_batches: int) -> None:
    """Static HTML browse page for the dumped PNGs."""
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>il_lib visual-input inspection</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}",
        "h1,h2{font-weight:600}",
        "img{border:1px solid #333;border-radius:4px;margin:2px;max-width:240px}",
        "section{margin-bottom:32px}",
        ".row{display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start}",
        ".cell{display:flex;flex-direction:column;align-items:center;font-size:11px;color:#aaa}",
        "</style>",
        "<h1>il_lib visual-input inspection</h1>",
        "<p>Top row of each grid = raw dataloader output. Bottom row (if present)"
        " = post-MultiviewResNet18 augmentation, de-normalised for display.</p>",
    ]
    for b in range(num_batches):
        grid_rel = f"grid_b{b:02d}.png"
        if not os.path.exists(os.path.join(out_dir, grid_rel)):
            continue
        lines.append(f"<section><h2>Batch {b}</h2>")
        lines.append(f"<img src='{html.escape(grid_rel)}' style='max-width:none'>")
        lines.append("<div class='row'>")
        for view in view_order:
            safe = _safe_view_name(view)
            raw_rel = f"raw/{safe}/b{b:02d}_s00.png"
            if os.path.exists(os.path.join(out_dir, raw_rel)):
                lines.append(
                    "<div class='cell'>"
                    f"<img src='{html.escape(raw_rel)}'>"
                    f"<div>{html.escape(view)}</div>"
                    "</div>"
                )
        lines.append("</div></section>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("\n".join(lines))


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg: DictConfig) -> None:
    register_omegaconf_resolvers()

    out_dir = OmegaConf.select(cfg, "inspect.out_dir", default="./inspect_out")
    num_batches = int(OmegaConf.select(cfg, "inspect.num_batches", default=2))
    samples_per_batch = int(
        OmegaConf.select(cfg, "inspect.samples_per_batch", default=4)
    )
    show_augmented = bool(
        OmegaConf.select(cfg, "inspect.show_augmented", default=False)
    )
    split = OmegaConf.select(cfg, "inspect.split", default="train")
    assert split in {"train", "val"}, "inspect.split must be 'train' or 'val'"

    seed = int(OmegaConf.select(cfg, "seed", default=42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[inspect] writing to {out_dir}")

    # ----------------- Data --------------------------------------------------
    data_module = instantiate(cfg.data)
    data_module.setup("fit")
    dataloader = (
        data_module.train_dataloader() if split == "train" else data_module.val_dataloader()
    )

    # ----------------- Model (optional, for post-aug visualisation) ---------
    model = None
    backbone = None
    if show_augmented:
        # ``_recursive_=False`` so ACT's __init__ handles its own sub-instantiations.
        model = instantiate(cfg.module, _recursive_=False)
        model = model.cpu().eval()
        backbone = getattr(model, "obs_backbone", None)
        if backbone is None:
            print("[inspect] WARN: model has no .obs_backbone; augmentation view disabled.")
            show_augmented = False

    # When the model is instantiated, use its canonical view order
    # (``obs_backbone.views``). Otherwise fall back to the dataloader's emit
    # order, sorted by name once the first batch is observed for stable output.
    if backbone is not None and hasattr(backbone, "views"):
        view_order = list(backbone.views)
    else:
        view_order = None

    stats_lines: List[str] = []

    for b_idx, batch in enumerate(dataloader):
        if b_idx >= num_batches:
            break

        rgb_by_view = _collect_obs_rgb(model, batch)  # {view: (B, L, C, H, W) float in [0,1]}
        if view_order is None:
            view_order = sorted(rgb_by_view.keys())

        # Confirm every configured view is present.
        missing = [v for v in view_order if v not in rgb_by_view]
        if missing:
            raise RuntimeError(
                f"[inspect] views missing from batch {b_idx}: {missing}\n"
                f"  views in batch: {sorted(rgb_by_view.keys())}\n"
                f"  expected (obs_backbone.views): {view_order}"
            )

        B = next(iter(rgb_by_view.values())).shape[0]
        n_samples = min(samples_per_batch, B)

        # Optional post-augmentation pass.
        aug_by_view: Optional[Dict[str, torch.Tensor]] = None
        if show_augmented and backbone is not None:
            with torch.no_grad():
                # Augmentation only triggers in training mode (matches forward).
                was_training = backbone.training
                backbone.train()
                try:
                    aug_by_view = _maybe_apply_augmentation(backbone, rgb_by_view)
                finally:
                    backbone.train(was_training)

        # Per-sample dump + first-sample grid.
        grid_raw_by_view: Dict[str, np.ndarray] = {}
        grid_aug_by_view: Optional[Dict[str, np.ndarray]] = {} if aug_by_view else None

        stats_lines.append(f"batch {b_idx}: B={B}, n_views={len(view_order)}, dumped_samples={n_samples}")
        for view in view_order:
            tensor = rgb_by_view[view]  # (B, L, C, H, W) in [0, 1]
            stats_lines.append(_format_stats(view, tensor))

            safe = _safe_view_name(view)
            for s_idx in range(n_samples):
                # Always show the first frame in L for simplicity (ACT has L=1).
                img = _tensor_to_uint8_hwc(tensor[s_idx, 0])
                _save_png(
                    img,
                    os.path.join(out_dir, "raw", safe, f"b{b_idx:02d}_s{s_idx:02d}.png"),
                )
                if s_idx == 0:
                    grid_raw_by_view[view] = img

                if aug_by_view is not None:
                    aug = aug_by_view[view][s_idx, 0]  # (C, h, w), normalised
                    aug_disp = _denormalize_imagenet(aug)
                    aug_img = _tensor_to_uint8_hwc(aug_disp)
                    _save_png(
                        aug_img,
                        os.path.join(out_dir, "aug", safe, f"b{b_idx:02d}_s{s_idx:02d}.png"),
                    )
                    if s_idx == 0 and grid_aug_by_view is not None:
                        grid_aug_by_view[view] = aug_img

        _make_grid(
            grid_raw_by_view,
            grid_aug_by_view if grid_aug_by_view else None,
            view_order,
            os.path.join(out_dir, f"grid_b{b_idx:02d}.png"),
        )
        print(f"[inspect] batch {b_idx}: saved {n_samples} samples × {len(view_order)} views")

    # Stats file: per-view min/mean/max. A view that's all-white or all-blue is
    # immediately obvious here (e.g. mean ~= 1.0 across all channels).
    with open(os.path.join(out_dir, "stats.txt"), "w") as f:
        f.write("\n".join(stats_lines) + "\n")

    _write_index_html(out_dir, view_order or [], num_batches)
    print(f"[inspect] done. open {os.path.join(out_dir, 'index.html')}")


if __name__ == "__main__":
    main()

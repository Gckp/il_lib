"""Quantify a trained ACT policy's sensitivity to its goal-image conditioning.

Diagnostic for "the policy appears to ignore the goal image" failures, where
``inspect_visual_inputs.py`` has already confirmed that the inputs to the
model look correct. The script loads a trained checkpoint, samples paired
validation demos, and runs two probes:

  1. Encoder probe. Compares the per-view feature maps produced by
     ``MultiviewResNet18`` for the same head observation paired with two
     different goal images. A model whose visual encoder distinguishes
     different goals should produce visibly different goal-view features.

  2. Action probe. Compares the action chunks predicted by the full ACT
     decoder for the same head observation paired with two different goal
     images, and also for two different head observations paired with the
     same goal image. The ratio of L1(actions | goal swap) to
     L1(actions | head swap) quantifies how much the policy attends to the
     goal relative to the current observation.

Interpretation:

  * Encoder distance high and action sensitivity high -> goal-conditioning
    is working end-to-end.
  * Encoder distance high but action sensitivity ~ 0 -> encoder distinguishes
    goals, but the decoder ignores them. Architectural goal-conditioning
    issue rather than encoder issue.
  * Encoder distance ~ 0 -> encoder is collapsing distinct goals to similar
    features. The downstream action collapse is then a *consequence*, not the
    primary failure.

The latent variable in ``ACT.forward`` is set to zeros at inference (see
``ACT.forward``'s ``else: latent_sample = torch.zeros(...)`` branch), so the
forward pass is deterministic given fixed inputs and no averaging over
latent samples is required.

Outputs (under ``+sensitivity.out_dir``):
    sensitivity_summary.txt    Per-pair metrics plus aggregate mean / std,
                               plus a one-line verdict.
    sensitivity_histogram.png  Histograms of L1 action sensitivity for each
                               swap type (goal swap, head swap, proprio swap
                               if applicable). Requires matplotlib; skipped
                               with a warning if matplotlib is unavailable.
    encoder_distance.png       Histograms of L2 / cosine distance between
                               encoder features for paired demos.
    pairs/{idx:03d}.png        First few swap pairs as side-by-side PNGs
                               (head | real goal | swap goal) so the visual
                               difference between paired goals can be
                               eyeballed.

Usage mirrors ``train.py``:

    python inspect_goal_sensitivity.py --config-name base_config_goal_image \\
        arch=act_goal task=behavior task.name=<task_name> robot=r1pro \\
        data_dir=<abs path> goal_image_project_root=<abs path> \\
        ckpt_path=<abs path to .pth> \\
        +sensitivity.out_dir=./sensitivity_out \\
        +sensitivity.num_pairs=64 \\
        use_wandb=false online_eval=null

Note on single-goal tasks: when every demonstration in the validation set
shares the same goal image, the goal-swap probe is structurally
uninformative (the "swapped" goal is identical to the original). The script
detects this case and prints a warning so the result is not misread as
goal-blindness in the model.
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

from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_torch


# Same de-normalization constants used by ``MultiviewResNet18``'s
# ``ResNet18_Weights.DEFAULT.transforms()`` preset.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Helpers: filesystem-safe view names and PNG writing
# ---------------------------------------------------------------------------


def _safe_view_name(view: str) -> str:
    """Make a view name filesystem-safe (cameras use ``::`` and ``:``)."""
    return view.replace("::", "__").replace(":", "_").replace("/", "_")


def _tensor_to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """Convert a ``(C, H, W)`` float tensor in [0, 1] to ``(H, W, C)`` uint8."""
    img = img_chw.detach().clamp(0.0, 1.0).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _save_png(arr_hwc_uint8: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_hwc_uint8).save(path)


# ---------------------------------------------------------------------------
# Helpers: batch / tensor manipulation
# ---------------------------------------------------------------------------


def _to_device(value: Any, device: torch.device) -> Any:
    """Move a nested tensor structure to ``device`` without changing layout."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


def _roll_nested(value: Any, shifts: int, dims: int = 0) -> Any:
    """Recursively ``torch.roll`` every tensor in a nested structure.

    Used to build "swapped" obs structures where one input (goal, head,
    proprio) is replaced with the next batch position's value while every
    other input is kept fixed.
    """
    if torch.is_tensor(value):
        return torch.roll(value, shifts=shifts, dims=dims)
    if isinstance(value, dict):
        return {k: _roll_nested(v, shifts, dims) for k, v in value.items()}
    if isinstance(value, list):
        return [_roll_nested(v, shifts, dims) for v in value]
    return value


def _detect_obs_key_groups(
    obs_keys: List[str], goal_view_name: Optional[str]
) -> Tuple[List[str], List[str], List[str]]:
    """Partition the raw obs-dict keys into (goal_rgb_keys, head_rgb_keys, proprio_keys).

    ``goal_rgb_keys`` is the subset of rgb keys belonging to the goal view;
    everything else with ``rgb`` in the key name is treated as a head view.
    ``proprio_keys`` is every top-level obs key that is not an image or a
    depth channel and that holds tensor-valued proprioception (qpos, eef,
    odom, ...). Visual modalities other than rgb (e.g. ``depth_linear``,
    ``seg_instance_id``) are excluded from all three groups; they go through
    the model unchanged.
    """
    goal_keys: List[str] = []
    head_keys: List[str] = []
    proprio_keys: List[str] = []

    goal_view_match = None
    if goal_view_name is not None:
        goal_view_match = f"{goal_view_name}::rgb"

    for k in obs_keys:
        if "rgb" in k:
            if goal_view_match is not None and k == goal_view_match:
                goal_keys.append(k)
            elif goal_view_name is None and "goal" in k:
                goal_keys.append(k)
            else:
                head_keys.append(k)
        elif "depth" in k or "seg_instance_id" in k or "pcd" in k:
            continue
        else:
            proprio_keys.append(k)
    return goal_keys, head_keys, proprio_keys


def _resolve_goal_view_name(cfg: DictConfig, backbone: Any) -> Optional[str]:
    """Return the goal view name from config, or infer it from backbone views.

    Returns ``None`` if no goal view can be identified, which surfaces the
    fact that the script was pointed at a non-goal-conditioned config.
    """
    cfg_name = OmegaConf.select(cfg, "goal_view_name", default=None)
    if cfg_name is not None:
        return str(cfg_name)
    if backbone is not None and hasattr(backbone, "views"):
        for v in backbone.views:
            if "goal" in v:
                return v
    return None


# ---------------------------------------------------------------------------
# Helpers: feature / action distance metrics
# ---------------------------------------------------------------------------


def _flatten_per_sample(t: torch.Tensor) -> torch.Tensor:
    """``(B, ...) -> (B, D)`` reshape, preserving batch dim."""
    return t.reshape(t.shape[0], -1)


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 distance for ``(B, D)`` tensors."""
    return torch.linalg.vector_norm(a - b, ord=2, dim=1)


def _pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity for ``(B, D)`` tensors."""
    return F.cosine_similarity(a, b, dim=1, eps=1e-8)


def _per_sample_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean per-sample L1 over all non-batch dims; returns ``(B,)``."""
    diff = (a - b).reshape(a.shape[0], -1).abs()
    return diff.mean(dim=1)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_views(model, processed_obs: dict) -> Dict[str, torch.Tensor]:
    """Run only the visual backbone and return the per-view feature dict.

    ``MultiviewResNet18`` with ``return_last_spatial_map=True`` returns a
    dict mapping view name to ``(B * L, C, h, w)``. For ACT this is ``L = 1``
    so the batch dimension is just ``B``.
    """
    backbone = model.obs_backbone
    if model._use_depth:
        rgb_dict = processed_obs["rgbd"]
    else:
        rgb_dict = processed_obs["rgb"]
    return backbone(rgb_dict)


@torch.no_grad()
def _predict_actions(model, processed_obs: dict) -> torch.Tensor:
    """Run the full ACT forward pass at inference and return ``(B, T_A, A)``.

    ``actions=None`` selects the inference path in ``ACT.forward``, which
    sets ``latent_sample`` to a deterministic zero tensor and produces a
    deterministic action chunk per input.
    """
    a_hat, _ = model.forward(obs=processed_obs, actions=None, is_pad=None)
    return a_hat


# ---------------------------------------------------------------------------
# Rendering / reporting
# ---------------------------------------------------------------------------


def _make_pair_collage(
    head_img: np.ndarray,
    goal_real_img: np.ndarray,
    goal_swap_img: np.ndarray,
    out_path: str,
) -> None:
    """Save a side-by-side PNG of (head obs, real goal, swap goal)."""
    ref_h, ref_w = head_img.shape[:2]
    cells = []
    for cell in [head_img, goal_real_img, goal_swap_img]:
        if cell.shape[:2] != (ref_h, ref_w):
            cell = np.array(Image.fromarray(cell).resize((ref_w, ref_h)))
        cells.append(cell)
    collage = np.concatenate(cells, axis=1)
    _save_png(collage, out_path)


def _maybe_write_histograms(
    out_dir: str,
    action_distances: Dict[str, np.ndarray],
    encoder_l2: Dict[str, np.ndarray],
    encoder_cos: Dict[str, np.ndarray],
) -> None:
    """Save histogram PNGs of the per-pair action / encoder metrics.

    Skipped with a warning when matplotlib is not importable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sensitivity] matplotlib not available; skipping histograms.")
        return

    # Action L1 histograms, one curve per swap type.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in action_distances.items():
        if vals.size == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.5, label=f"{name}  (mean={vals.mean():.4f})")
    ax.set_xlabel("Per-sample L1 distance between predicted action chunks")
    ax.set_ylabel("Number of demo pairs")
    ax.set_title("Action sensitivity to input swaps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_histogram.png"), dpi=120)
    plt.close(fig)

    # Encoder distance histograms, one curve per view (typically head + goal).
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    for view, vals in encoder_l2.items():
        if vals.size == 0:
            continue
        ax[0].hist(
            vals, bins=30, alpha=0.5, label=f"{view}  (mean={vals.mean():.4f})"
        )
    ax[0].set_xlabel("L2 distance between per-view encoder features (real vs goal swap)")
    ax[0].set_ylabel("Number of demo pairs")
    ax[0].legend(fontsize=8)
    for view, vals in encoder_cos.items():
        if vals.size == 0:
            continue
        ax[1].hist(
            vals, bins=30, alpha=0.5, label=f"{view}  (mean={vals.mean():.4f})"
        )
    ax[1].set_xlabel("Cosine similarity between per-view encoder features (real vs goal swap)")
    ax[1].set_ylabel("Number of demo pairs")
    ax[1].legend(fontsize=8)
    fig.suptitle("Encoder sensitivity to goal swap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "encoder_distance.png"), dpi=120)
    plt.close(fig)


def _format_array(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:<4d} mean={arr.mean():.4f} std={arr.std():.4f} "
        f"median={np.median(arr):.4f} min={arr.min():.4f} max={arr.max():.4f}"
    )


def _write_summary(
    out_dir: str,
    action_distances: Dict[str, np.ndarray],
    encoder_l2: Dict[str, np.ndarray],
    encoder_cos: Dict[str, np.ndarray],
    head_unchanged_l2: Dict[str, np.ndarray],
    goal_view_name: Optional[str],
    same_goal_pairs: int,
    total_pairs: int,
    notes: List[str],
) -> str:
    """Write ``sensitivity_summary.txt`` and return the one-line verdict."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("ACT goal-image sensitivity report")
    lines.append("=" * 78)
    lines.append(f"goal view name : {goal_view_name}")
    lines.append(f"demo pairs     : {total_pairs}")
    if same_goal_pairs > 0:
        lines.append(
            f"pairs with identical goal images: {same_goal_pairs}/{total_pairs} "
            "(goal-swap is uninformative for these)"
        )

    lines.append("")
    lines.append("Action sensitivity (per-sample L1 distance in normalized action space)")
    lines.append("-" * 78)
    for name, vals in action_distances.items():
        lines.append(f"  {name:<24s}  {_format_array(vals)}")

    # Ratio of goal sensitivity to head-obs sensitivity (the key diagnostic).
    verdict = "(insufficient data for verdict)"
    if (
        "goal_swap" in action_distances
        and "head_swap" in action_distances
        and action_distances["goal_swap"].size > 0
        and action_distances["head_swap"].size > 0
    ):
        eps = 1e-8
        ratios = action_distances["goal_swap"] / (action_distances["head_swap"] + eps)
        lines.append("")
        lines.append("Goal-vs-head sensitivity ratio (per-pair)")
        lines.append("-" * 78)
        lines.append(f"  goal_swap / head_swap   {_format_array(ratios)}")
        mean_ratio = float(ratios.mean())
        if mean_ratio < 0.05:
            verdict = (
                f"action sensitivity to goal swap / head swap = "
                f"{mean_ratio:.3f}  ->  goal conditioning is COLLAPSED at the decoder"
            )
        elif mean_ratio < 0.25:
            verdict = (
                f"action sensitivity to goal swap / head swap = "
                f"{mean_ratio:.3f}  ->  goal influences actions only weakly"
            )
        else:
            verdict = (
                f"action sensitivity to goal swap / head swap = "
                f"{mean_ratio:.3f}  ->  goal is being used by the decoder"
            )

    lines.append("")
    lines.append("Encoder sensitivity to goal swap (L2 distance between feature maps)")
    lines.append("-" * 78)
    for view, vals in encoder_l2.items():
        lines.append(f"  L2   {view:<60s} {_format_array(vals)}")
    for view, vals in encoder_cos.items():
        lines.append(f"  cos  {view:<60s} {_format_array(vals)}")

    if head_unchanged_l2:
        lines.append("")
        lines.append("Sanity check: head-view encoder features should be unchanged under goal swap")
        lines.append("-" * 78)
        for view, vals in head_unchanged_l2.items():
            lines.append(f"  L2   {view:<60s} {_format_array(vals)}")

    if notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-" * 78)
        for note in notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append("Verdict")
    lines.append("-" * 78)
    lines.append(f"  {verdict}")

    out_path = os.path.join(out_dir, "sensitivity_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg: DictConfig) -> None:
    register_omegaconf_resolvers()

    ckpt_path = OmegaConf.select(cfg, "ckpt_path", default=None)
    if not ckpt_path:
        raise RuntimeError(
            "ckpt_path must be set (e.g. ckpt_path=/abs/path/to/step.pth)."
        )

    out_dir = os.path.abspath(
        OmegaConf.select(cfg, "sensitivity.out_dir", default="./sensitivity_out")
    )
    num_pairs = int(OmegaConf.select(cfg, "sensitivity.num_pairs", default=64))
    max_pair_images = int(
        OmegaConf.select(cfg, "sensitivity.max_pair_images", default=8)
    )
    split = OmegaConf.select(cfg, "sensitivity.split", default="val")
    assert split in {"train", "val"}, "sensitivity.split must be 'train' or 'val'"

    seed = int(OmegaConf.select(cfg, "seed", default=42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[sensitivity] writing to {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sensitivity] device: {device}")

    # ----------------- Data -------------------------------------------------
    data_module = instantiate(cfg.data)
    data_module.setup("fit")
    dataloader = (
        data_module.train_dataloader() if split == "train" else data_module.val_dataloader()
    )

    # ----------------- Model + checkpoint -----------------------------------
    model = instantiate(cfg.module, _recursive_=False)
    ckpt = load_torch(ckpt_path)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[sensitivity] WARN: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[sensitivity] WARN: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model = model.to(device).eval()

    goal_view_name = _resolve_goal_view_name(cfg, getattr(model, "obs_backbone", None))
    if goal_view_name is None:
        raise RuntimeError(
            "Could not resolve a goal view. Set ``goal_view_name`` in the "
            "config or use an arch that includes a view containing 'goal'."
        )
    print(f"[sensitivity] goal view: {goal_view_name}")

    # ----------------- Probe loop ------------------------------------------
    action_distances: Dict[str, List[float]] = {
        "goal_swap": [],
        "head_swap": [],
        "proprio_swap": [],
    }
    encoder_l2_goal: Dict[str, List[float]] = {}
    encoder_cos_goal: Dict[str, List[float]] = {}
    head_unchanged_l2: Dict[str, List[float]] = {}
    notes: List[str] = []
    pair_images_written = 0
    same_goal_pairs = 0
    total_pairs = 0

    for batch_idx, batch in enumerate(dataloader):
        if total_pairs >= num_pairs:
            break

        batch = _to_device(batch, device)
        obs_keys = list(batch["obs"].keys())
        goal_keys, head_keys, proprio_keys = _detect_obs_key_groups(
            obs_keys, goal_view_name
        )
        if not goal_keys:
            raise RuntimeError(
                f"No goal rgb key matching '{goal_view_name}::rgb' in obs keys: {obs_keys}"
            )
        if batch_idx == 0:
            print(
                f"[sensitivity] head rgb keys ({len(head_keys)}): {head_keys}"
            )
            print(f"[sensitivity] goal rgb keys ({len(goal_keys)}): {goal_keys}")
            print(f"[sensitivity] proprio keys: {proprio_keys}")

        # Roll-by-1 pairing: position i ends up paired with position (i+1) mod B.
        # That gives a clean, deterministic per-batch pair structure where every
        # row of the batch contributes exactly one pair.
        def _roll_keys(obs: dict, keys: List[str]) -> dict:
            new_obs = dict(obs)
            for k in keys:
                new_obs[k] = _roll_nested(new_obs[k], shifts=1, dims=0)
            return new_obs

        obs_real = dict(batch["obs"])
        obs_goal_swap = _roll_keys(obs_real, goal_keys)
        obs_head_swap = _roll_keys(obs_real, head_keys)
        obs_proprio_swap = (
            _roll_keys(obs_real, proprio_keys) if proprio_keys else None
        )

        def _processed(obs: dict) -> dict:
            return model.process_data({"obs": obs}, extract_action=False)

        proc_real = _processed(obs_real)
        proc_goal_swap = _processed(obs_goal_swap)
        proc_head_swap = _processed(obs_head_swap)
        proc_proprio_swap = (
            _processed(obs_proprio_swap) if obs_proprio_swap is not None else None
        )

        # --- Encoder probe --------------------------------------------------
        feats_real = _encode_views(model, proc_real)
        feats_goal_swap = _encode_views(model, proc_goal_swap)
        for view, t_real in feats_real.items():
            t_swap = feats_goal_swap[view]
            flat_real = _flatten_per_sample(t_real)
            flat_swap = _flatten_per_sample(t_swap)
            l2 = _pairwise_l2(flat_real, flat_swap).cpu().numpy()
            cos = _pairwise_cosine(flat_real, flat_swap).cpu().numpy()
            encoder_l2_goal.setdefault(view, []).extend(l2.tolist())
            encoder_cos_goal.setdefault(view, []).extend(cos.tolist())
            if "goal" not in view:
                # Sanity: head-view features under goal swap should be unchanged
                # (the head image was not modified). Any non-zero distance here
                # is a smoking gun for state leaking across views in the backbone.
                head_unchanged_l2.setdefault(view, []).extend(l2.tolist())

        # --- Action probe ---------------------------------------------------
        a_real = _predict_actions(model, proc_real)
        a_goal_swap = _predict_actions(model, proc_goal_swap)
        a_head_swap = _predict_actions(model, proc_head_swap)

        action_distances["goal_swap"].extend(
            _per_sample_l1(a_real, a_goal_swap).cpu().numpy().tolist()
        )
        action_distances["head_swap"].extend(
            _per_sample_l1(a_real, a_head_swap).cpu().numpy().tolist()
        )

        if proc_proprio_swap is not None:
            a_proprio_swap = _predict_actions(model, proc_proprio_swap)
            action_distances["proprio_swap"].extend(
                _per_sample_l1(a_real, a_proprio_swap).cpu().numpy().tolist()
            )

        # --- Single-goal detection ----------------------------------------
        # If every demo in the batch shares the same goal image, the goal-swap
        # probe is structurally uninformative because the "swapped" goal is
        # identical to the original. Count such pairs so the summary can flag
        # the situation explicitly.
        gk = goal_keys[0]
        goal_tensor = batch["obs"][gk]
        rolled = torch.roll(goal_tensor, shifts=1, dims=0)
        equal_mask = torch.all(
            (goal_tensor == rolled).reshape(goal_tensor.shape[0], -1), dim=1
        )
        same_goal_pairs += int(equal_mask.sum().item())

        # --- Pair PNG dumps ------------------------------------------------
        # Save a few side-by-side collages to make the goal swap visually
        # concrete in the report.
        rgb_real = proc_real["rgb"] if "rgb" in proc_real else proc_real["rgbd"]
        rgb_swap = (
            proc_goal_swap["rgb"] if "rgb" in proc_goal_swap else proc_goal_swap["rgbd"]
        )
        head_view = head_keys[0].rsplit("::", 1)[0] if head_keys else None
        goal_view = goal_keys[0].rsplit("::", 1)[0]
        for b in range(goal_tensor.shape[0]):
            if pair_images_written >= max_pair_images:
                break
            if head_view is None:
                break
            head_img = _tensor_to_uint8_hwc(rgb_real[head_view][b, 0])
            goal_real_img = _tensor_to_uint8_hwc(rgb_real[goal_view][b, 0])
            goal_swap_img = _tensor_to_uint8_hwc(rgb_swap[goal_view][b, 0])
            _make_pair_collage(
                head_img,
                goal_real_img,
                goal_swap_img,
                os.path.join(out_dir, "pairs", f"{pair_images_written:03d}.png"),
            )
            pair_images_written += 1

        total_pairs += goal_tensor.shape[0]
        print(
            f"[sensitivity] batch {batch_idx}: total_pairs={total_pairs} "
            f"(target {num_pairs})"
        )

    if same_goal_pairs == total_pairs and total_pairs > 0:
        notes.append(
            "All sampled demo pairs share the same goal image. This dataset "
            "appears to be single-goal, so the goal-swap probe is structurally "
            "uninformative and ~0 action sensitivity is expected. Run on the "
            "goal-perturbed variant to obtain a meaningful goal-sensitivity "
            "measurement."
        )

    # ----------------- Aggregate + report -----------------------------------
    np_action_distances = {
        k: np.asarray(v, dtype=np.float32) for k, v in action_distances.items()
    }
    np_encoder_l2 = {
        k: np.asarray(v, dtype=np.float32) for k, v in encoder_l2_goal.items()
    }
    np_encoder_cos = {
        k: np.asarray(v, dtype=np.float32) for k, v in encoder_cos_goal.items()
    }
    np_head_unchanged_l2 = {
        k: np.asarray(v, dtype=np.float32) for k, v in head_unchanged_l2.items()
    }

    verdict = _write_summary(
        out_dir,
        np_action_distances,
        np_encoder_l2,
        np_encoder_cos,
        np_head_unchanged_l2,
        goal_view_name,
        same_goal_pairs,
        total_pairs,
        notes,
    )

    _maybe_write_histograms(
        out_dir, np_action_distances, np_encoder_l2, np_encoder_cos
    )

    print()
    print(f"[sensitivity] verdict: {verdict}")
    print(f"[sensitivity] full report: {os.path.join(out_dir, 'sensitivity_summary.txt')}")


if __name__ == "__main__":
    main()

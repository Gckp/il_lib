import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import tree
from il_lib.utils.functional_utils import implements_method
from il_lib.utils.tree_utils import tree_value_at_path
from il_lib.utils.file_utils import f_join
from typing import Union, List, Tuple


def accelerator_requests_cuda(accelerator: str) -> bool:
    """True if the Lightning accelerator setting may use CUDA."""
    a = str(accelerator).lower()
    if a in ("gpu", "cuda"):
        return True
    if a == "auto":
        return torch.cuda.is_available()
    return False


def wait_for_cuda_devices_ready(enabled: bool = True) -> None:
    """Block until the local process can run real work on its CUDA device.

    On shared clusters, SLURM may start the job before the GPU is fully released
    from a prior tenant; NCCL then fails during ``init_process_group`` with
    "CUDA-capable device(s) is/are busy or unavailable". This loop polls until
    a small allocation and kernel succeed or the timeout is reached.

    Environment (optional):

    - ``IL_LIB_CUDA_WAIT``: set to ``0`` / ``false`` / ``no`` to disable.
    - ``IL_LIB_CUDA_WAIT_TIMEOUT_SEC``: max seconds to wait (default ``14400``).
    - ``IL_LIB_CUDA_WAIT_POLL_SEC``: sleep between attempts (default ``15``).
    """
    if not enabled:
        return
    flag = os.environ.get("IL_LIB_CUDA_WAIT", "1").lower()
    if flag in ("0", "false", "no", "off"):
        return
    if not torch.cuda.is_available():
        return

    timeout_sec = float(os.environ.get("IL_LIB_CUDA_WAIT_TIMEOUT_SEC", "14400"))
    poll_sec = float(os.environ.get("IL_LIB_CUDA_WAIT_POLL_SEC", "15"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    wait_started = time.monotonic()
    deadline = wait_started + timeout_sec
    attempt = 0

    def _probe() -> None:
        torch.cuda.set_device(local_rank)
        torch.cuda.init()
        # Allocation + GEMM exercises the driver similarly to NCCL setup.
        t = torch.empty((256, 256), device=f"cuda:{local_rank}", dtype=torch.float32)
        torch.mm(t, t)
        torch.cuda.synchronize()
        del t

    while True:
        try:
            _probe()
            if attempt > 0:
                print(
                    f"[il_lib] CUDA local_rank={local_rank} ready after "
                    f"{time.monotonic() - wait_started:.0f}s.",
                    flush=True,
                )
            return
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err:
                raise
            retryable = any(
                s in err
                for s in (
                    "busy",
                    "unavailable",
                    "not ready",
                    "device not initialized",
                    "unknown cuda error",
                )
            )
            if not retryable:
                raise
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"CUDA device local_rank={local_rank} still busy or unusable after "
                f"{timeout_sec:.0f}s (set IL_LIB_CUDA_WAIT_TIMEOUT_SEC to wait longer, "
                "or IL_LIB_CUDA_WAIT=0 to skip this wait)."
            ) from None
        attempt += 1
        if attempt == 1 or attempt % max(1, int(60 / max(poll_sec, 1))) == 0:
            print(
                f"[il_lib] Waiting for CUDA local_rank={local_rank} to become usable "
                f"(attempt {attempt}, {max(0.0, deadline - time.monotonic()):.0f}s left); "
                f"sleeping {poll_sec:.1f}s.",
                flush=True,
            )
        time.sleep(poll_sec)


def seed_everywhere(seed, torch_deterministic=False, rank=0):
    """set seed across modules"""
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def sequential_split_dataset(dataset: torch.utils.data.Dataset, split_portions: list[float]):
    """
    Split a dataset into multiple datasets, each with a different portion of the
    original dataset. Uses torch.utils.data.Subset.
    """
    from il_lib.utils.functional_utils import accumulate

    assert len(split_portions) > 0, "split_portions must be a non-empty list"
    assert all(0.0 <= p <= 1.0 for p in split_portions), f"{split_portions=}"
    assert abs(sum(split_portions) - 1.0) < 1e-6, f"{sum(split_portions)=} != 1.0"
    L = len(dataset)
    assert L > 0, "dataset must be non-empty"
    # split the list with proportions
    lengths = [int(p * L) for p in split_portions]
    # make sure the last split fills the full dataset
    lengths[-1] += L - sum(lengths)
    indices = list(range(L))

    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def load_torch(*fpath: str, map_location="cpu") -> dict:
    """
    Default maps to "cpu"
    """
    fpath = str(f_join(fpath))
    try:
        return torch.load(fpath, map_location=map_location, weights_only=False)
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n\n --- Error loading {fpath}")



def set_requires_grad(model, requires_grad):
    if torch.is_tensor(model):
        model.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad



def freeze_params(model):
    set_requires_grad(model, False)
    if not torch.is_tensor(model):
        model.eval()


def unfreeze_params(model):
    set_requires_grad(model, True)
    if not torch.is_tensor(model):
        model.train()


def classify_accuracy(
    output,
    target,
    topk: Union[int, List[int], Tuple[int]] = 1,
    mask=None,
    reduction="mean",
    scale_100=False,
):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0

    Args:
        topk: if int, return a single acc. If tuple, return a tuple of accs
        mask: shape [batch_size,], binary mask of whether to include this sample or not
    """
    if isinstance(topk, int):
        topk = [topk]
        is_int = True
    else:
        is_int = False

    batch_size = target.size(0)
    assert output.size(0) == batch_size
    if mask is not None:
        assert mask.dim() == 1
        assert mask.size(0) == batch_size

    assert reduction in ["sum", "mean", "none"]
    if reduction != "mean":
        assert not scale_100, f"reduce={reduction} does not support scale_100=True"

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if mask is not None:
            correct = mask * correct

        mult = 100.0 if scale_100 else 1.0
        res = []
        for k in topk:
            correct_k = correct[:k].int().sum(dim=0)
            if reduction == "mean":
                if mask is not None:
                    # fmt: off
                    res.append(
                        float(correct_k.float().sum().mul_(mult / (mask.sum().item() + 1e-6)).item())
                    )
                    # fmt: on
                else:
                    res.append(
                        float(correct_k.float().sum().mul_(mult / batch_size).item())
                    )
            elif reduction == "sum":
                res.append(int(correct_k.sum().item()))
            elif reduction == "none":
                res.append(correct_k)
            else:
                raise NotImplementedError(f"Unknown reduce={reduction}")

    if is_int:
        assert len(res) == 1, "INTERNAL"
        return res[0]
    else:
        return res


def load_state_dict(objects, states, strip_prefix=None, strict=False):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        if not implements_method(obj, "load_state_dict"):
            raise ValueError(
                f"Object {type(obj)} does not support load_state_dict() method"
            )
        try:
            state = tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {
                k[len(strip_prefix) :]: v
                for k, v in state.items()
                if k.startswith(strip_prefix)
            }
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    return tree.map_structure_with_path(_load, objects)

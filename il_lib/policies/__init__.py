"""Policy implementations for ``il_lib``.

Imports are lazy (PEP 562) so that ``hydra.utils.get_class`` /
``'_target_: il_lib.policies.ACT'`` only pulls in ACT and its dependencies,
not diffusion / WBVIMA stacks that may drag extra packages.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "ACT",
    "BC_RNN",
    "DiffusionPolicy",
    "WBVIMA",
]

if TYPE_CHECKING:
    from .act_policy import ACT as ACT
    from .bcrnn_policy import BC_RNN as BC_RNN
    from .diffusion_policy import DiffusionPolicy as DiffusionPolicy
    from .wbvima_policy import WBVIMA as WBVIMA


def __getattr__(name: str) -> Any:
    if name == "ACT":
        from .act_policy import ACT

        return ACT
    if name == "BC_RNN":
        from .bcrnn_policy import BC_RNN

        return BC_RNN
    if name == "DiffusionPolicy":
        from .diffusion_policy import DiffusionPolicy

        return DiffusionPolicy
    if name == "WBVIMA":
        from .wbvima_policy import WBVIMA

        return WBVIMA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

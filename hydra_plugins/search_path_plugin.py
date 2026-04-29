"""Hydra search-path plugin for il_lib.

Primary goal: make task / robot configs resolve from the OmniGibson checkout
paired with the current il_lib workspace, even if the active Python env
imports an ``omnigibson`` package from a different clone.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from pathlib import Path
import omnigibson as og
iiil = None
try:
    import iiil
except ImportError:
    pass

class SearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Prefer the OmniGibson configs that live next to this il_lib checkout.
        # layout: <repo>/il/il_lib/hydra_plugins/search_path_plugin.py
        #      -> <repo>/OmniGibson/omnigibson/learning/configs
        repo_root_cfg = (
            Path(__file__).resolve().parents[3]
            / "OmniGibson"
            / "omnigibson"
            / "learning"
            / "configs"
        )
        if repo_root_cfg.exists():
            search_path.append("il_lib_local", str(repo_root_cfg))

        # Fallback to whichever omnigibson package is importable in this env.
        search_path.append("il_lib_pkg", f"{og.__path__[0]}/learning/configs")
        if iiil is not None:
            search_path.append("iiil", f"{iiil.__path__[0]}/configs")

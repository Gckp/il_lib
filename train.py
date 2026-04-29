"""Hydra entrypoint for il_lib training.

Python 3.14+ argparse validates ``help=`` with ``in``/``not in`` on the raw
object; Hydra 1.3.x passes ``LazyCompletionHelp`` for ``--shell-completion``,
which breaks before ``@hydra.main`` runs. Coerce non-str help until Hydra
ships a fixed release (facebookresearch/hydra#3121).
"""
import argparse
import copy
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

import hydra

from il_lib.utils.training_utils import seed_everywhere
from il_lib.utils.config_utils import omegaconf_to_dict
from il_lib.training import Trainer


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg):
    cfg.seed = seed_everywhere(cfg.seed)
    trainer_ = Trainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(omegaconf_to_dict(cfg))
    trainer_.fit()
    trainer_.test()


if __name__ == "__main__":
    main()

"""Load v4 data_loader and v5 load_odds. Top-level to avoid src namespace conflict."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


def _clear_src():
    for k in list(sys.modules.keys()):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]


# v4 data_loader
_clear_src()
sys.path.insert(0, str(_root / "v4"))
from src.data_loader import load_data, get_team_info

# v5 load_odds
_clear_src()
sys.path.insert(0, str(_root / "v5"))
from src.load_odds import load_odds, apply_power_debias, fit_market_mapping, market_prob_learned

# v8 first for subsequent imports
_clear_src()
sys.path.insert(0, str(_root / "v8"))

__all__ = [
    "load_data",
    "get_team_info",
    "load_odds",
    "apply_power_debias",
    "fit_market_mapping",
    "market_prob_learned",
]

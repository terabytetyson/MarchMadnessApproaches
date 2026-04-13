"""V10 deps: load_data from v4, load_kalshi from v10."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


def _clear_src():
    for k in list(sys.modules.keys()):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]


# v4 data_loader for get_team_info if needed
_clear_src()
sys.path.insert(0, str(_root / "v4"))
from src.data_loader import load_data, get_team_info

# v10
_clear_src()
sys.path.insert(0, str(_root / "v10"))

__all__ = ["load_data", "get_team_info"]

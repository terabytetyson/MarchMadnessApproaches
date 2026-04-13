"""Configuration for v9: Classic Paris (box + style) + market odds. No BERT. Men's + women's."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = PROJECT_ROOT.parent.parent if "iterations" in str(PROJECT_ROOT) else PROJECT_ROOT
DATA_DIR = os.environ.get("DATA_DIR", str(_REPO_ROOT / "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(_REPO_ROOT / "models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(_REPO_ROOT))

_default_odds = _REPO_ROOT / "data" / "cbb_round1_odds_all_champions_2002_2025.json"
ODDS_PATH = os.environ.get(
    "ODDS_PATH",
    str(_default_odds if _default_odds.exists() else Path.home() / "Downloads" / "cbb_round1_odds_all_champions_2002_2025.json"),
)

TRAIN_START = 2003
TRAIN_END = 2024

# Women's detailed results (WRegularSeasonDetailedResults) start in 2010
WOMEN_DETAILED_START = 2010

VAL_RATIO = 0.2
TEST_YEARS = [2018, 2025]
VAL_YEARS = [2018]

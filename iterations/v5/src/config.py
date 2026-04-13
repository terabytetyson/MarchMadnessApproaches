"""Configuration for v5: market odds baseline + residual prediction on style features."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = PROJECT_ROOT.parent.parent if "iterations" in str(PROJECT_ROOT) else PROJECT_ROOT
DATA_DIR = os.environ.get("DATA_DIR", str(_REPO_ROOT / "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(_REPO_ROOT / "models"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(_REPO_ROOT))

# Odds file (pre-tournament championship odds)
ODDS_PATH = os.environ.get("ODDS_PATH", str(Path.home() / "Downloads" / "cbb_round1_odds_all_champions_2002_2025.json"))

TRAIN_START = 2003  # Men's; odds data starts 2002
TRAIN_END = 2024
# v5 is men-only (odds data is men's)
GENDER = "men"

VAL_RATIO = 0.2
TEST_YEARS = [2018, 2025]
VAL_YEARS = [2018]

# Market mapping: calibrate raw sigmoid(c*(r1-r2)) on training data
CALIBRATE_MARKET = True

# Residual model: Ridge (L2) or XGBoost
RESIDUAL_MODEL = "ridge"  # "ridge" or "xgb"
RIDGE_ALPHA = 1.0
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4

"""Configuration for v10: goto_conversion + simulate n brackets (Kalshi progression probs)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = PROJECT_ROOT.parent.parent if "iterations" in str(PROJECT_ROOT) else PROJECT_ROOT
DATA_DIR = os.environ.get("DATA_DIR", str(_REPO_ROOT / "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(_REPO_ROOT))
BRACKET_APP_PUBLIC = _REPO_ROOT / "bracket-app" / "public"

# Kalshi progression probabilities JSON (R32, S16, E8, F4, F2, Champion as 0-100)
# Prefer project data/kalshi_cbb_tournament_odds_2026.json; fallback to Downloads
_default_kalshi = _REPO_ROOT / "data" / "kalshi_cbb_tournament_odds_2026.json"
if not _default_kalshi.exists():
    _default_kalshi = Path.home() / "Downloads" / "kalshi_cbb_tournament_odds_2026.json"
KALSHI_PATH = os.environ.get("KALSHI_PATH", str(_default_kalshi))

# Robinhood men's odds (cents format)
ROBINHOOD_PATH = os.environ.get(
    "ROBINHOOD_PATH",
    str(Path.home() / "Downloads" / "mens_college_basketball_2026_odds.json"),
)

KALSHI_WCBB_PATH = os.environ.get(
    "KALSHI_WCBB_PATH",
    str(Path.home() / "Downloads" / "kalshi_wcbb_tournament_odds_2026.json"),
)

# Season for submission (must match bracket in MNCAATourneySeeds)
SEASON = int(os.environ.get("SEASON", "2026"))

# Number of bracket simulations for aggregation
N_SIMS = int(os.environ.get("N_SIMS", "50000"))

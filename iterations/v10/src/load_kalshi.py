"""Load Kalshi progression probabilities and map to TeamIDs."""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

# Kalshi round keys (exclude R64 - not needed for bracket sim)
ROUND_KEYS = ["R32", "S16", "E8", "F4", "F2", "Champion"]

# Round index for monotonicity: R32=0, S16=1, ..., Champion=5
ROUND_ORDER = {k: i for i, k in enumerate(ROUND_KEYS)}


def _normalize_name(name: str) -> str:
    """Normalize team name for matching: lowercase, collapse spaces, strip punctuation."""
    n = str(name).strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = n.replace(".", "").replace("'", "'").replace("`", "'")
    return n


def _build_name_to_id(data_dir: Optional[str] = None) -> dict:
    """Build mapping: normalized_name -> TeamID (men's 1000-2000)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    teams = pd.read_csv(data_dir / "MTeams.csv")
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv")

    name_to_id = {}
    for _, row in teams.iterrows():
        tid = int(row["TeamID"])
        if 1000 <= tid < 2000:
            name = _normalize_name(row["TeamName"])
            if name and name not in name_to_id:
                name_to_id[name] = tid

    for _, row in spellings.iterrows():
        tid = int(row["TeamID"])
        if 1000 <= tid < 2000:
            name = _normalize_name(row["TeamNameSpelling"])
            if name and name not in name_to_id:
                name_to_id[name] = tid

    # Kalshi-specific aliases
    aliases = [
        ("st johns", "st john's"),
        ("st johns (ny)", "st john's"),
        ("iowa st", "iowa state"),
        ("michigan st", "michigan state"),
        ("ohio st", "ohio state"),
        ("wright st", "wright state"),
        ("north dakota st", "north dakota state"),
        ("kennesaw st", "kennesaw state"),
        ("saint marys", "saint mary's"),
        ("saint louis", "st louis"),
        ("miami fl", "miami (fl)"),
        ("miami oh", "miami (oh)"),
        ("miami - florida", "miami (fl)"),
        ("miami-florida", "miami (fl)"),
        ("texas a&m", "texas a&m"),
        ("queens university", "queens (nc)"),
        ("california baptist", "cal baptist"),
        ("liu", "long island"),
    ]
    for alias, canonical in aliases:
        a_norm = _normalize_name(alias)
        c_norm = _normalize_name(canonical)
        if c_norm in name_to_id and a_norm not in name_to_id:
            name_to_id[a_norm] = name_to_id[c_norm]

    return name_to_id


def _enforce_monotonicity(probs: dict) -> dict:
    """Ensure R32 >= S16 >= E8 >= F4 >= F2 >= Champion."""
    out = dict(probs)
    prev = 100.0
    for k in ROUND_KEYS:
        v = out.get(k, 0)
        v = min(v, prev)
        v = max(0, v)
        out[k] = v
        prev = v
    return out


def load_kalshi_progression(
    kalshi_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> tuple[dict, dict]:
    """Load Kalshi JSON and return (team_id -> {round: prob}, unmapped_names).

    Probs are 0-1. Enforces monotonicity. R64 is skipped.
    Returns: (progression_lookup, unmapped_names).
    progression_lookup: (team_id) -> {"R32": 0.83, "S16": 0.53, ...}
    """
    kalshi_path = kalshi_path or config.KALSHI_PATH
    data_dir = data_dir or config.DATA_DIR

    with open(kalshi_path) as f:
        raw = json.load(f)

    name_to_id = _build_name_to_id(data_dir)
    progression = {}
    unmapped = []

    for team_name, rounds in raw.get("teams", {}).items():
        norm = _normalize_name(team_name)
        tid = name_to_id.get(norm)
        if tid is None:
            unmapped.append(team_name)
            continue

        pct = {k: rounds.get(k, 0) / 100.0 for k in ROUND_KEYS}
        pct = _enforce_monotonicity(pct)
        progression[tid] = pct

    return progression, unmapped

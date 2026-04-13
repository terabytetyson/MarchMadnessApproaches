"""Bracket structure and per-game win probabilities via goto_conversion."""

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .load_kalshi import ROUND_KEYS, load_kalshi_progression

_EPS = 1e-9

# Game round (1-6) -> (current_round_key, next_round_key) for P(win) = next/current
ROUND_PROGRESSION = [
    ("R32", None),        # R1 (R64 games): P(win) = R32
    ("R32", "S16"),       # R2 (R32 games)
    ("S16", "E8"),       # R3 (S16)
    ("E8", "F4"),        # R4 (E8)
    ("F4", "F2"),        # R5 (F4)
    ("F2", "Champion"),  # R6 (Champ)
]


def _get_seed_to_team(season: int, data_dir: Optional[str] = None, gender: str = "M") -> dict:
    """seed_str -> TeamID. gender: 'M' or 'W'."""
    data_dir = Path(data_dir or config.DATA_DIR)
    prefix = "M" if gender == "M" else "W"
    seeds = pd.read_csv(data_dir / f"{prefix}NCAATourneySeeds.csv")
    subset = seeds[seeds["Season"] == season]
    return dict(zip(subset["Seed"], subset["TeamID"]))


def get_slots_ordered(season: int, data_dir: Optional[str] = None, gender: str = "M") -> list:
    """Slots in simulation order: play-ins first, then R1, R2, ..., R6. gender: 'M' or 'W'."""
    data_dir = Path(data_dir or config.DATA_DIR)
    prefix = "M" if gender == "M" else "W"
    slots = pd.read_csv(data_dir / f"{prefix}NCAATourneySlots.csv")
    subset = slots[slots["Season"] == season]
    rows = list(zip(subset["Slot"], subset["StrongSeed"], subset["WeakSeed"]))
    # Play-in slots (don't start with R1-R6) must come before R1 that references them
    playins = [(s, a, b) for s, a, b in rows if not (s.startswith("R1") or s.startswith("R2") or s.startswith("R3") or s.startswith("R4") or s.startswith("R5") or s.startswith("R6"))]
    rest = [(s, a, b) for s, a, b in rows if s.startswith("R1") or s.startswith("R2") or s.startswith("R3") or s.startswith("R4") or s.startswith("R5") or s.startswith("R6")]
    return playins + rest


def _get_game_round(slot: str) -> int:
    """Slot -> round 1-6 (0 for play-in)."""
    if slot.startswith("R1"):
        return 1
    if slot.startswith("R2"):
        return 2
    if slot.startswith("R3"):
        return 3
    if slot.startswith("R4"):
        return 4
    if slot.startswith("R5"):
        return 5
    if slot.startswith("R6"):
        return 6
    return 0


def get_match_prob(
    team1_id: int,
    team2_id: int,
    round_num: int,
    progression: dict,
) -> float:
    """P(team1 beats team2) in given round, using goto_conversion."""
    try:
        import goto_conversion
    except ImportError:
        raise ImportError("pip install goto-conversion")

    # Play-in (round 0): use R32 as P(advance to R64)
    if round_num == 0:
        round_num = 1  # treat like R1 for progression

    prog1 = progression.get(team1_id, {})
    prog2 = progression.get(team2_id, {})

    curr_key, next_key = ROUND_PROGRESSION[round_num - 1]
    if next_key is None:
        p1_raw = prog1.get("R32", 0.5)
        p2_raw = prog2.get("R32", 0.5)
    else:
        curr1 = max(prog1.get(curr_key, 0), _EPS)
        next1 = prog1.get(next_key, 0)
        curr2 = max(prog2.get(curr_key, 0), _EPS)
        next2 = prog2.get(next_key, 0)

        def _win_prob(curr, next_, prog):
            if curr > 0 and next_ < curr:
                return next_ / curr
            # Flat/inverted from monotonicity capping: use previous round's ratio
            if round_num >= 2:
                prev_curr_key, prev_next_key = ROUND_PROGRESSION[round_num - 2]
                pc, pn = prog.get(prev_curr_key, 0), prog.get(prev_next_key, 0)
                if pc > _EPS and pn < pc:
                    return pn / pc
            return 0.5

        p1_raw = _win_prob(curr1, next1, prog1)
        p2_raw = _win_prob(curr2, next2, prog2)

    p1_raw = max(min(p1_raw, 1 - _EPS), _EPS)
    p2_raw = max(min(p2_raw, 1 - _EPS), _EPS)

    inv_odds = [1.0 / p1_raw, 1.0 / p2_raw]
    probs = goto_conversion.goto_conversion(
        inv_odds,
        multiplicativeIfImprudentOdds=True,
    )
    return float(probs[0])

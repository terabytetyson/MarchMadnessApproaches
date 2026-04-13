"""Blending odds-only (goto) with Kalshi progression. Weight w by round, seeds, meet_prob."""

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .bracket import _get_game_round, _get_seed_to_team, get_match_prob


def _build_slot_to_seeds(season: int, data_dir: Optional[str], gender: str = "M") -> dict:
    """slot -> set of seed strings that can reach this slot. Recursively expands."""
    data_dir = Path(data_dir or config.DATA_DIR)
    prefix = "M" if gender == "M" else "W"
    slots_df = pd.read_csv(data_dir / f"{prefix}NCAATourneySlots.csv")
    subset = slots_df[slots_df["Season"] == season]
    slot_pairs = [(r["Slot"], r["StrongSeed"], r["WeakSeed"]) for _, r in subset.iterrows()]

    # All seed strings (direct + play-in)
    all_seeds = set()
    for _, s, w in slot_pairs:
        all_seeds.add(s)
        all_seeds.add(w)

    cache = {}

    def seeds_for(s: str) -> set:
        if s in cache:
            return cache[s]
        # Direct seed (W01, X16, W11a, etc.)
        if s in all_seeds and not s.startswith("R"):
            cache[s] = {s}
            return cache[s]
        # Slot reference (R1W1, etc.) - find the slot and expand
        for slot, strong, weak in slot_pairs:
            if slot == s:
                cache[s] = seeds_for(strong) | seeds_for(weak)
                return cache[s]
        cache[s] = {s}
        return cache[s]

    result = {}
    for slot, strong, weak in slot_pairs:
        result[slot] = seeds_for(strong) | seeds_for(weak)
    return result


def _build_seed_pair_to_round(season: int, data_dir: Optional[str], gender: str = "M") -> dict:
    """(seed1, seed2) -> first round (0-6) they could meet. 0=play-in. Normalize seed order."""
    slot_to_seeds = _build_slot_to_seeds(season, data_dir, gender)
    round_to_slots = {}
    for slot, seeds in slot_to_seeds.items():
        r = _get_game_round(slot)
        round_to_slots.setdefault(r, []).append(seeds)

    result = {}
    for r in range(0, 7):
        for slot_seeds in round_to_slots.get(r, []):
            for s1 in slot_seeds:
                for s2 in slot_seeds:
                    if s1 != s2:
                        key = (min(s1, s2), max(s1, s2))
                        if key not in result:
                            result[key] = r
    return result


def _parse_seed_num(seed_str: str) -> int:
    """W01 -> 1, W16a -> 16, X11 -> 11."""
    import re
    s = str(seed_str).replace("a", "").replace("b", "")
    m = re.search(r"\d{1,2}", s)
    return int(m.group(0)) if m else 99


def _prog_reach_round(team_id: int, round_num: int, progression: dict) -> float:
    """P(team reaches this round) from progression. round 0=play-in use R32, 1=R32, 2=S16, 3=E8, 4=F4, 5=F2, 6=Champion."""
    keys = ["R32", "S16", "E8", "F4", "F2", "Champion"]
    if round_num < 0:
        return 0.5
    r = max(0, min(round_num, 5))
    prog = progression.get(team_id, {})
    return prog.get(keys[r], 0.5)


def compute_blend_weight(
    round_num: int,
    seed1: int,
    seed2: int,
    meet_prob: float,
) -> float:
    """Compute w for P_final = (1-w)*P_goto + w*P_kalshi. R1: 50/50. R2-4: 30% Kalshi. F4/Champ: 70% Kalshi."""
    # Play-in (round 0): treat as round 1 for weight
    if round_num == 0:
        round_num = 1
    if round_num == 1:
        return 0.5  # 50/50 blend: Kalshi and sportsbook (P_goto)

    # Rounds 2+: no Kalshi if either seed > 12
    if seed1 > 12 or seed2 > 12:
        return 0.0

    seed_gap = abs(seed1 - seed2)

    if round_num == 2:
        w = 0.30
    elif round_num == 3:
        w = 0.30
    elif round_num == 4:
        w = 0.30
    else:  # 5 or 6 (F4, Championship)
        w = 0.70

    if seed_gap > 4:
        w *= 0.5
    elif seed_gap <= 2:
        w *= 1.15

    if meet_prob < 0.10:
        w *= 0.5
    if meet_prob < 0.03:
        w = 0.0

    cap = 0.70 if round_num >= 5 else 0.30
    return min(w, cap)

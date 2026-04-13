"""Seed-only market model (no Paris, no odds). For bottom seeds 13-16."""

import re
from typing import Optional

_EPS = 1e-9


def _parse_seed_num(seed_str: str) -> int:
    """W01 -> 1, W16a -> 16, X11 -> 11."""
    s = str(seed_str).replace("a", "").replace("b", "")
    m = re.search(r"\d{1,2}", s)
    return int(m.group(0)) if m else 0


def _seed_to_prior(seed: int) -> float:
    """Seed-based prior: (17-seed)/136 so seed 1 -> 16/136, seed 16 -> 1/136."""
    seed = max(1, min(16, int(seed)))
    return (17 - seed) / 136.0


def market_prob_seed_only(seed1: int, seed2: int) -> float:
    """P(team1 beats team2) using seed-only prior. Bradley-Terry: prior1/(prior1+prior2)."""
    p1 = _seed_to_prior(seed1)
    p2 = _seed_to_prior(seed2)
    return float(p1 / (p1 + p2 + _EPS))


def market_prob_odds(team1: int, team2: int, odds: dict) -> Optional[float]:
    """P(team1 beats team2) from championship odds. Bradley-Terry: p1/(p1+p2).
    Returns None if either team missing from odds."""
    p1 = odds.get(team1)
    p2 = odds.get(team2)
    if p1 is None or p2 is None or (p1 + p2) < _EPS:
        return None
    return float(p1 / (p1 + p2 + _EPS))


def market_prob_odds_goto_conversion(team1: int, team2: int, odds: dict) -> Optional[float]:
    """P(team1 beats team2) from championship odds via goto_conversion.
    Returns None if either team missing from odds."""
    try:
        import goto_conversion
    except ImportError:
        raise ImportError("pip install goto-conversion")

    p1 = odds.get(team1)
    p2 = odds.get(team2)
    if p1 is None or p2 is None or p1 < _EPS or p2 < _EPS:
        return None

    inv_odds = [1.0 / max(p1, _EPS), 1.0 / max(p2, _EPS)]
    probs = goto_conversion.goto_conversion(
        inv_odds,
        multiplicativeIfImprudentOdds=True,
    )
    return float(probs[0])


def market_prob_for_13_16(
    team1: int,
    team2: int,
    seed1: int,
    seed2: int,
    odds: dict,
) -> float:
    """For 13-16 games: use championship odds if both teams in odds, else seed prior."""
    if not use_market_seed_for_pair(seed1, seed2):
        return market_prob_seed_only(seed1, seed2)
    prob = market_prob_odds(team1, team2, odds)
    if prob is not None:
        return prob
    return market_prob_seed_only(seed1, seed2)


def use_market_seed_for_pair(seed1: int, seed2: int) -> bool:
    """Use seed-only market model when either team is seed 13-16."""
    return seed1 >= 13 or seed2 >= 13

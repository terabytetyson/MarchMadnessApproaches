"""Simulate n brackets and aggregate to matchup probabilities."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from . import config
from .bracket import _get_seed_to_team, get_slots_ordered, _get_game_round, get_match_prob
from .load_kalshi import load_kalshi_progression


def simulate_brackets(
    season: int,
    progression: dict,
    n_sims: int,
    data_dir: Optional[str] = None,
    seed: int = 42,
    gender: str = "M",
) -> dict:
    """Run n_sims bracket simulations. Return (team1_id, team2_id) -> P(team1 wins).

    team1_id < team2_id always (for submission format). gender: 'M' or 'W'.
    """
    data_dir = data_dir or config.DATA_DIR
    seed_to_team = _get_seed_to_team(season, data_dir, gender)
    slots = get_slots_ordered(season, data_dir, gender)

    rng = random.Random(seed)

    # matchup -> (wins_by_lower, total_occurrences)
    matchup_counts = defaultdict(lambda: [0, 0])

    for _ in tqdm(range(n_sims), desc="Simulating brackets"):
        bracket_state = {}  # slot/seed -> winner TeamID

        for slot, strong_seed, weak_seed in slots:
            round_num = _get_game_round(slot)
            if round_num == 0:
                # Play-in
                t1 = seed_to_team.get(strong_seed)
                t2 = seed_to_team.get(weak_seed)
            else:
                t1 = bracket_state.get(strong_seed) or seed_to_team.get(strong_seed)
                t2 = bracket_state.get(weak_seed) or seed_to_team.get(weak_seed)

            if t1 is None or t2 is None:
                winner = t1 or t2
            else:
                p = get_match_prob(t1, t2, round_num, progression)
                winner = t1 if rng.random() < p else t2

                # Record matchup for aggregation
                team_lo, team_hi = min(t1, t2), max(t1, t2)
                key = (team_lo, team_hi)
                matchup_counts[key][1] += 1
                if winner == team_lo:
                    matchup_counts[key][0] += 1

            bracket_state[slot] = winner

    # Convert to probabilities
    result = {}
    for (t_lo, t_hi), (wins, total) in matchup_counts.items():
        result[(t_lo, t_hi)] = wins / total if total > 0 else 0.5

    return result

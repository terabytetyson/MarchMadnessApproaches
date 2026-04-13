"""Build v8 training data: Paris text (box + style) + market odds. Men's only."""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import deps
from deps import load_data, get_team_info, load_odds, apply_power_debias, fit_market_mapping, market_prob_learned

from src import config
from src.text_builder import matchup_to_text, compute_matchup_stats_paris


def build_training_data(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    exclude_seasons: Optional[list] = None,
    include_only_seasons: Optional[list] = None,
    debiased_lookup: Optional[dict] = None,
    c: Optional[float] = None,
    intercept: float = 0.0,
) -> pd.DataFrame:
    """Build DataFrame with Paris text + market_prob. Men's only."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir)
    seed_map = data.get("seed_map", {})

    if not odds_lookup:
        raise FileNotFoundError(f"Odds not found at {odds_path}. V8 requires odds.")

    if debiased_lookup is None or c is None:
        games_for_fit = []
        tourney = data["m_tourney"]
        subset = tourney[
            (tourney["Season"] >= config.TRAIN_START) & (tourney["Season"] <= config.TRAIN_END)
        ]
        if exclude_seasons:
            subset = subset[~subset["Season"].isin(exclude_seasons)]
        for _, row in subset.iterrows():
            w_id, l_id = row["WTeamID"], row["LTeamID"]
            team1_id, team2_id = min(w_id, l_id), max(w_id, l_id)
            actual = 1.0 if w_id == team1_id else 0.0
            games_for_fit.append((int(row["Season"]), team1_id, team2_id, actual))
        alpha, c, intercept = fit_market_mapping(odds_lookup, seed_map, games_for_fit)
        debiased_lookup = apply_power_debias(odds_lookup, alpha)

    exclude = set(exclude_seasons or [])
    include_only = set(include_only_seasons or [])

    tourney_df = data["m_tourney"]
    subset = tourney_df[
        (tourney_df["Season"] >= config.TRAIN_START) & (tourney_df["Season"] <= config.TRAIN_END)
    ]
    if include_only:
        subset = subset[subset["Season"].isin(include_only)]
    else:
        subset = subset[~subset["Season"].isin(exclude)]

    rows = []
    for _, row in subset.iterrows():
        season = int(row["Season"])
        daynum = int(row["DayNum"])
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]
        w_score = row["WScore"]
        l_score = row["LScore"]

        team1_id = min(w_id, l_id)
        team2_id = max(w_id, l_id)
        margin_raw = w_score - l_score
        margin = margin_raw if w_id == team1_id else -margin_raw

        team1_info = get_team_info(data, team1_id, season, "men")
        team2_info = get_team_info(data, team2_id, season, "men")

        loc_key = (season, daynum, w_id, l_id)
        loc_alt = (season, daynum, l_id, w_id)
        game_loc = data["game_location"].get(loc_key) or data["game_location"].get(loc_alt)
        location = f"{game_loc[0]}, {game_loc[1]}" if game_loc else "Neutral site"

        matchup_stats = compute_matchup_stats_paris(team1_info, team2_info)
        s1, s2 = team1_info["seed"], team2_info["seed"]
        seed_vs_seed = data["seed_vs_seed"].get((min(s1, s2), max(s1, s2))) if data.get("seed_vs_seed") else None

        market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)

        text = matchup_to_text(
            team1_info, team2_info, season, "men",
            location=location,
            seed_vs_seed=seed_vs_seed,
            matchup_stats=matchup_stats,
            market_prob=market_prob,
        )
        rows.append({
            "text": text,
            "margin": margin,
            "season": season,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": team1_info["team_name"],
            "team2_name": team2_info["team_name"],
            "winner_id": w_id,
            "gender": "men",
        })

        # Swapped order
        matchup_stats_swapped = compute_matchup_stats_paris(team2_info, team1_info)
        text_swapped = matchup_to_text(
            team2_info, team1_info, season, "men",
            location=location,
            seed_vs_seed=seed_vs_seed,
            matchup_stats=matchup_stats_swapped,
            market_prob=1.0 - market_prob,
        )
        rows.append({
            "text": text_swapped,
            "margin": -margin,
            "season": season,
            "team1_id": team2_id,
            "team2_id": team1_id,
            "team1_name": team2_info["team_name"],
            "team2_name": team1_info["team_name"],
            "winner_id": w_id,
            "gender": "men",
        })

    return pd.DataFrame(rows)


def print_samples(df: pd.DataFrame, n: int = 3):
    """Print n sample training examples."""
    samples = df.sample(min(n, len(df))) if len(df) >= n else df
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        winner = row["team1_name"] if row["margin"] > 0 else row["team2_name"]
        print(f"\n{'='*60}")
        print(f"=== Sample {i} (season {int(row['season'])}, margin={row['margin']:.1f}) ===")
        print(f"Winner: {winner}")
        print(f"{'='*60}")
        print(row["text"])
        print()

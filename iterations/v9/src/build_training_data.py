"""Build training data for v9: Paris features + market odds. Men's + women's. No BERT."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import deps
from deps import load_data, get_team_info, load_odds, apply_power_debias, fit_market_mapping, market_prob_learned

from src import config
from src.features import get_paris_features, FEATURE_ORDER


def _build_gender_data(
    data: dict,
    tourney_df: pd.DataFrame,
    gender: str,
    debiased_lookup: Optional[dict],
    c: Optional[float],
    intercept: float,
    seed_map: dict,
    exclude: set,
    include_only: set,
) -> list:
    """Build rows for one gender. Women use market_prob=0.5 (no odds)."""
    start = config.TRAIN_START
    if gender == "women":
        # Women's detailed results (box stats) start in 2010
        start = max(start, getattr(config, "WOMEN_DETAILED_START", 2010))
    subset = tourney_df[
        (tourney_df["Season"] >= start) & (tourney_df["Season"] <= config.TRAIN_END)
    ]
    if include_only:
        subset = subset[subset["Season"].isin(include_only)]
    else:
        subset = subset[~subset["Season"].isin(exclude)]

    rows = []
    for _, row in subset.iterrows():
        season = int(row["Season"])
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]

        team1_id = min(w_id, l_id)
        team2_id = max(w_id, l_id)
        label = 1 if w_id == team1_id else 0

        team1_info = get_team_info(data, team1_id, season, gender)
        team2_info = get_team_info(data, team2_id, season, gender)

        if gender == "men" and debiased_lookup is not None and c is not None:
            market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)
        else:
            market_prob = 0.5  # women: no odds, use neutral

        feats = get_paris_features(team1_info, team2_info, market_prob=market_prob)
        rows.append({
            "features": feats,
            "label": label,
            "season": season,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "gender": gender,
        })

        feats_swapped = get_paris_features(team2_info, team1_info, market_prob=1.0 - market_prob)
        rows.append({
            "features": feats_swapped,
            "label": 1 - label,
            "season": season,
            "team1_id": team2_id,
            "team2_id": team1_id,
            "gender": gender,
        })

    return rows


def build_training_data(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    exclude_seasons: Optional[list] = None,
    include_only_seasons: Optional[list] = None,
    debiased_lookup: Optional[dict] = None,
    c: Optional[float] = None,
    intercept: float = 0.0,
    include_women: bool = True,
) -> pd.DataFrame:
    """Build DataFrame with Paris features + market_prob, label. Men's + women's."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir) if odds_path else {}
    seed_map = data.get("seed_map", {})

    if not odds_lookup and include_women:
        pass  # ok, women use market_prob=0.5
    elif not odds_lookup:
        raise FileNotFoundError(f"Odds not found at {odds_path}. Men's training requires odds.")

    if debiased_lookup is None or c is None:
        if odds_lookup:
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
        else:
            debiased_lookup, c, intercept = {}, 1.0, 0.0

    exclude = set(exclude_seasons or [])
    include_only = set(include_only_seasons or [])

    rows = _build_gender_data(
        data, data["m_tourney"], "men",
        debiased_lookup, c, intercept, seed_map, exclude, include_only,
    )
    if include_women:
        rows.extend(_build_gender_data(
            data, data["w_tourney"], "women",
            None, None, 0.0, seed_map, exclude, include_only,
        ))

    return pd.DataFrame(rows)


def print_samples(df: pd.DataFrame, n: int = 3):
    """Print n sample rows."""
    samples = df.sample(min(n, len(df))) if len(df) >= n else df
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        gender = row.get("gender", "men")
        print(f"\n--- Sample {i} (season {int(row['season'])}, {gender}, label={int(row['label'])}) ---")
        print(f"  features: {dict(zip(FEATURE_ORDER, [f'{x:.2f}' for x in row['features']]))}")

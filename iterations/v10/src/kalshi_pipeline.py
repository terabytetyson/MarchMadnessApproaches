"""Men's Kalshi: simulate brackets from progression odds, fill gaps with goto_conversion / 13–16 odds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .blend import _build_seed_pair_to_round
from .bracket import get_match_prob
from .load_championship_odds import load_championship_odds
from .load_kalshi import load_kalshi_progression
from .market_seed import _parse_seed_num, market_prob_for_13_16, use_market_seed_for_pair
from .simulate import simulate_brackets


def generate_kalshi_submission(
    kalshi_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    season: Optional[int] = None,
    n_sims: Optional[int] = None,
    output_path: Optional[str] = None,
    stage: str = "Stage1",
) -> str:
    """Monte Carlo over bracket structure + Kalshi progression; 13–16 from title odds; else goto fallback."""
    kalshi_path = kalshi_path or config.KALSHI_PATH
    data_dir = data_dir or config.DATA_DIR
    season = season or config.SEASON
    n_sims = n_sims if n_sims is not None else config.N_SIMS
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "output_kalshi_mens.csv")

    odds = load_championship_odds(data_dir=data_dir)
    progression, _unmapped = load_kalshi_progression(kalshi_path, data_dir)

    seeds_path = Path(data_dir) / "MNCAATourneySeeds.csv"
    seeds = pd.read_csv(seeds_path)
    if season not in seeds["Season"].values:
        raise ValueError(
            f"Season {season} not in MNCAATourneySeeds. Add seeds for {season} or set SEASON."
        )

    season_seeds = seeds[seeds["Season"] == season]
    team_ids = sorted(season_seeds["TeamID"].unique().tolist())
    team_to_seed_str = dict(zip(season_seeds["TeamID"], season_seeds["Seed"]))
    seed_pair_to_round = _build_seed_pair_to_round(season, data_dir, "M")

    sim_probs = simulate_brackets(
        season, progression, n_sims, data_dir, seed=42, gender="M"
    )
    # sim_probs[(lo, hi)] = P(lower TeamID wins)

    rows = []
    for i, t1 in enumerate(team_ids):
        for t2 in team_ids[i + 1 :]:
            seed_str1 = team_to_seed_str.get(t1, "W16")
            seed_str2 = team_to_seed_str.get(t2, "W16")
            s1 = _parse_seed_num(seed_str1)
            s2 = _parse_seed_num(seed_str2)

            if use_market_seed_for_pair(s1, s2):
                pred = market_prob_for_13_16(t1, t2, s1, s2, odds)
            else:
                lo, hi = min(t1, t2), max(t1, t2)
                key = (lo, hi)
                if key in sim_probs:
                    pred = sim_probs[key]  # P(t1 wins) since t1 == lo
                else:
                    sk = (min(seed_str1, seed_str2), max(seed_str1, seed_str2))
                    round_num = seed_pair_to_round.get(sk, 1)
                    pred = get_match_prob(t1, t2, max(round_num, 1), progression)

            if pred > 0.96:
                pred = 1.0
            elif pred < 0.04:
                pred = 0.0

            rows.append({"ID": f"{season}_{t1}_{t2}", "Pred": pred})

    sub = pd.DataFrame(rows)

    sample_path = Path(data_dir) / f"SampleSubmission{stage}.csv"
    if not sample_path.exists():
        sample_path = Path(data_dir) / "SampleSubmissionStage1.csv"
    if sample_path.exists():
        full_sample = pd.read_csv(sample_path)
        season_in_sample = full_sample["ID"].str.startswith(f"{season}_").any()
        if season_in_sample:
            full_preds = {row["ID"]: 0.5 for _, row in full_sample.iterrows()}
            for _, row in sub.iterrows():
                full_preds[row["ID"]] = row["Pred"]
            full_sample["Pred"] = [full_preds[rid] for rid in full_sample["ID"]]
            full_sample.to_csv(output_path, index=False)
        else:
            sub.to_csv(output_path, index=False)
    else:
        sub.to_csv(output_path, index=False)

    print(f"Saved Kalshi men's submission to {output_path} ({len(sub)} tournament pairs, n_sims={n_sims})")
    return output_path

"""
features.py — Paris-style matchup text builder for v12 (DistilBERT).

Identical feature engineering to v11; kept self-contained so each
iteration is independently runnable without cross-directory imports.

Text format:
  "[S1] Duke vs [S16] Wagner | Massey gap: +42.3 | Round: Round of 64 | Seed gap: -15"

Numeric feature vector (2 dims):
  [seed_gap, massey_gap]  — fed to the numeric fusion branch in model.py
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


# ── data loading ─────────────────────────────────────────────────────────────

def _load_csv(data_dir: str, name: str, **kwargs) -> pd.DataFrame:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_and_prepare_data(data_dir: str):
    """
    Load Kaggle March Madness CSVs.
    Returns (results_df, seeds_df, teams_df, ordinals_df).
    """
    results = _load_csv(data_dir, "MNCAATourneyCompactResults.csv")

    seeds = _load_csv(data_dir, "MNCAATourneySeeds.csv")
    seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)

    teams = _load_csv(data_dir, "MTeams.csv")

    ordinals_parts = []
    for part in ["MMasseyOrdinals_part1.csv", "MMasseyOrdinals_part2.csv"]:
        p = os.path.join(data_dir, part)
        if os.path.exists(p):
            ordinals_parts.append(pd.read_csv(p))
    if not ordinals_parts:
        p = os.path.join(data_dir, "MMasseyOrdinals.csv")
        if os.path.exists(p):
            ordinals_parts.append(pd.read_csv(p))
    ordinals = pd.concat(ordinals_parts, ignore_index=True) if ordinals_parts else pd.DataFrame()

    return results, seeds, teams, ordinals


# ── Massey helper ─────────────────────────────────────────────────────────────

def _massey_ratings(ordinals_df: pd.DataFrame, season: int) -> dict:
    """Average ordinal rank per team, inverted so higher = better."""
    if ordinals_df.empty:
        return {}
    df = ordinals_df[ordinals_df["Season"] == season]
    if df.empty:
        return {}
    avg = df.groupby("TeamID")["OrdinalRank"].mean()
    return (avg.max() - avg + 1).to_dict()


# ── text construction ─────────────────────────────────────────────────────────

_ROUND_MAP = {1: "First Four", 2: "Round of 64", 3: "Round of 32",
              4: "Sweet 16", 5: "Elite 8", 6: "Final Four", 7: "Championship"}


def _build_one_text(
    team_a: str, seed_a: int,
    team_b: str, seed_b: int,
    massey_a: float, massey_b: float,
    round_num: Optional[int] = None,
) -> str:
    seed_str_a = f"[S{seed_a}]" if seed_a else ""
    seed_str_b = f"[S{seed_b}]" if seed_b else ""
    massey_gap = round(massey_a - massey_b, 1)
    gap_sign = "+" if massey_gap >= 0 else ""
    round_str = _ROUND_MAP.get(round_num, "Tournament") if round_num else "Tournament"

    parts = [
        f"{seed_str_a} {team_a} vs {seed_str_b} {team_b}",
        f"Massey gap: {gap_sign}{massey_gap}",
        f"Round: {round_str}",
        f"Seed gap: {seed_a - seed_b}",
    ]
    return " | ".join(p.strip() for p in parts)


def build_matchup_texts(
    results_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    ordinals_df: pd.DataFrame,
    split: str = "train",
    season: Optional[int] = None,
) -> Tuple[List[str], Optional[List[int]], np.ndarray, Optional[List[str]]]:
    """
    Build matchup texts + numeric features.

    Returns
    -------
    texts       : list[str]
    labels      : list[int] (1=first team won, 0=second) — None for test
    numeric     : ndarray (N, 2)  [seed_gap, massey_gap]
    matchup_ids : list[str] "SEASON_LOWTID_HIGHTID" — None for train
    """
    seed_lookup = {
        (int(r["Season"]), int(r["TeamID"])): int(r["SeedNum"])
        for _, r in seeds_df.iterrows()
    }
    name_lookup = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    texts, labels, numeric_rows, matchup_ids = [], [], [], []

    if split == "train":
        for _, row in results_df.iterrows():
            s = int(row["Season"])
            w, l = int(row["WTeamID"]), int(row["LTeamID"])
            massey = _massey_ratings(ordinals_df, s)

            seed_w = seed_lookup.get((s, w), 8)
            seed_l = seed_lookup.get((s, l), 8)
            mass_w = massey.get(w, 0.0)
            mass_l = massey.get(l, 0.0)

            # positive: winner listed first
            texts.append(_build_one_text(
                name_lookup.get(w, str(w)), seed_w,
                name_lookup.get(l, str(l)), seed_l,
                mass_w, mass_l, row.get("DayNum")
            ))
            labels.append(1)
            numeric_rows.append([seed_w - seed_l, mass_w - mass_l])

            # negative: loser listed first
            texts.append(_build_one_text(
                name_lookup.get(l, str(l)), seed_l,
                name_lookup.get(w, str(w)), seed_w,
                mass_l, mass_w, row.get("DayNum")
            ))
            labels.append(0)
            numeric_rows.append([seed_l - seed_w, mass_l - mass_w])

        return texts, labels, np.array(numeric_rows, dtype=np.float32), None

    else:  # test
        if season is None:
            raise ValueError("season must be provided for split='test'")
        massey = _massey_ratings(ordinals_df, season)
        s_seeds = seeds_df[seeds_df["Season"] == season]
        team_ids = sorted(s_seeds["TeamID"].unique())

        for i, tid_a in enumerate(team_ids):
            for tid_b in team_ids[i + 1:]:
                low_id, high_id = min(tid_a, tid_b), max(tid_a, tid_b)
                mid = f"{season}_{low_id}_{high_id}"

                seed_a = seed_lookup.get((season, tid_a), 8)
                seed_b = seed_lookup.get((season, tid_b), 8)
                mass_a = massey.get(tid_a, 0.0)
                mass_b = massey.get(tid_b, 0.0)

                texts.append(_build_one_text(
                    name_lookup.get(tid_a, str(tid_a)), seed_a,
                    name_lookup.get(tid_b, str(tid_b)), seed_b,
                    mass_a, mass_b,
                ))
                numeric_rows.append([seed_a - seed_b, mass_a - mass_b])
                matchup_ids.append(mid)

        return texts, None, np.array(numeric_rows, dtype=np.float32), matchup_ids

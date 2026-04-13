"""Load pre-tournament odds and convert to matchup probabilities."""

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config

_EPS = 1e-8


def _american_to_implied_prob(odds_str: str) -> float:
    """Convert American odds to implied probability. +325 -> 0.235, -200 -> 0.667."""
    s = str(odds_str).strip()
    if not s or s == "nan":
        return 0.5  # unknown
    m = re.match(r"([+-]?\d+)", s)
    if not m:
        return 0.5
    val = int(m.group(1))
    if val >= 0:
        return 100 / (val + 100)
    return abs(val) / (abs(val) + 100)


def _season_str_to_year(s: str) -> int:
    """'2024-2025' -> 2025, '2002-2003' -> 2003."""
    parts = str(s).split("-")
    if len(parts) >= 2:
        return int(parts[1])
    return int(parts[0]) if parts else 0


def _normalize_team_name(name: str) -> str:
    """Lowercase, strip, collapse spaces, remove periods for matching."""
    n = str(name).strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = n.replace(".", "").replace("'", "'")
    return n


def build_team_name_to_id(data_dir: Optional[str] = None) -> dict:
    """Build mapping: normalized_name -> TeamID (men's only, 1000-2000)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    teams = pd.read_csv(data_dir / "MTeams.csv")
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv")

    name_to_id = {}
    for _, row in teams.iterrows():
        tid = int(row["TeamID"])
        if 1000 <= tid < 2000:  # men's
            name = _normalize_team_name(row["TeamName"])
            if name and name not in name_to_id:
                name_to_id[name] = tid

    for _, row in spellings.iterrows():
        tid = int(row["TeamID"])
        if 1000 <= tid < 2000:
            name = _normalize_team_name(row["TeamNameSpelling"])
            if name and name not in name_to_id:
                name_to_id[name] = tid

    # Common variations from odds JSON (alias -> canonical in MTeams/MTeamSpellings)
    aliases = [
        ("st johns", "st john's"),
        ("nc wilmington", "unc wilmington"),
        ("unc ashville", "unc asheville"),
        ("troy state", "troy"),
        ("sam houston", "sam houston state"),
        ("central michigan", "c michigan"),
        ("miami - florida", "miami fl"),
        ("miami-florida", "miami fl"),
        # Unmapped names from odds sanity check
        ("binghampton", "binghamton"),
        ("louisiana lafayette", "louisiana"),
        ("miami, ohio", "miami oh"),
        ("southeast louisiana", "southeastern louisiana"),
        ("st peters", "st peter's"),
        ("texas a&m corpus christi", "texas a&m-corpus christi"),
        ("u mass", "massachusetts"),
        ("wintrop", "winthrop"),
    ]
    for alias, canonical in aliases:
        if canonical in name_to_id and alias not in name_to_id:
            name_to_id[alias] = name_to_id[canonical]

    return name_to_id


def load_odds(odds_path: Optional[str] = None, data_dir: Optional[str] = None) -> dict:
    """Load odds JSON and return (season_year, team_id) -> implied_prob.

    Also returns name_to_id for lookup. Teams not in odds get implied_prob = 0.5 (neutral).
    """
    odds_path = odds_path or config.ODDS_PATH
    if not os.path.exists(odds_path):
        return {}

    with open(odds_path) as f:
        raw = json.load(f)

    name_to_id = build_team_name_to_id(data_dir)
    result = {}

    for season_block in raw.get("seasons", []):
        season_str = season_block.get("season", "")
        year = _season_str_to_year(season_str)
        if year < 2002:
            continue

        for item in season_block.get("round1_odds", []):
            team_name = item.get("team", "")
            odds_str = item.get("round1_odds", "+10000")
            implied = _american_to_implied_prob(odds_str)

            norm = _normalize_team_name(team_name)
            tid = name_to_id.get(norm)
            if tid is not None:
                result[(year, tid)] = implied

    return result


def implied_to_rating(p: float) -> float:
    """Convert implied probability to log-odds (latent strength): r = log(p/(1-p))."""
    p = np.clip(float(p), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _seed_to_prior(seed: int) -> float:
    """Seed-based prior prob: (17-seed)/136 so seed 1 -> 16/136, seed 16 -> 1/136."""
    seed = max(1, min(16, int(seed)))
    return (17 - seed) / 136.0


def apply_power_debias(
    odds_lookup: dict,
    alpha: float,
) -> dict:
    """Apply power debias: p_i* = p_i^α, renormalize per season. Returns (season, team_id) -> p*."""
    from collections import defaultdict
    by_season = defaultdict(dict)
    for (season, tid), p in odds_lookup.items():
        p = np.clip(float(p), _EPS, 1 - _EPS)
        by_season[season][tid] = p ** alpha

    result = {}
    for season, team_probs in by_season.items():
        z = sum(team_probs.values())
        if z > 0:
            for tid, p in team_probs.items():
                result[(season, tid)] = p / z
    return result


_LOG_STRENGTH_CLIP = 12.0  # cap log(p*) to avoid overflow in LogisticRegression


def _get_log_debiased_strength(
    debiased_lookup: dict,
    season: int,
    team_id: int,
    seed_map: Optional[dict] = None,
) -> float:
    """log(p*) for matchup formula. Uses seed prior if team not in odds. Clipped to avoid overflow."""
    p_star = debiased_lookup.get((season, team_id))
    if p_star is not None:
        log_val = np.log(np.clip(p_star, _EPS, 1))
    elif seed_map:
        prior = _seed_to_prior(seed_map.get((season, team_id), 8))
        log_val = np.log(np.clip(prior, _EPS, 1))
    else:
        log_val = np.log(1 / 68.0)
    return float(np.clip(log_val, -_LOG_STRENGTH_CLIP, _LOG_STRENGTH_CLIP))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def market_prob_learned(
    debiased_lookup: dict,
    season: int,
    team1_id: int,
    team2_id: int,
    c: float,
    seed_map: Optional[dict] = None,
    intercept: float = 0.0,
) -> float:
    """P(team1 beats team2) = σ(intercept + c · (log p1* - log p2*)). Uses debiased probs + seed prior for missing."""
    log_p1 = _get_log_debiased_strength(debiased_lookup, season, team1_id, seed_map)
    log_p2 = _get_log_debiased_strength(debiased_lookup, season, team2_id, seed_map)
    return float(_sigmoid(intercept + c * (log_p1 - log_p2)))


def fit_alpha_and_c(
    odds_lookup: dict,
    seed_map: dict,
    games: list,
) -> tuple:
    """Fit α (power debias) via grid search; fit c and intercept via logistic regression.

    Massey-style: P(win) = σ(intercept + c·(s_i - s_j)) where s_i = log(p_i*).
    Select α by Brier on validation (competition metric). Use regularization (C=0.1).

    Each game: (season, team1_id, team2_id, actual).
    Returns (alpha, c, intercept).
    """
    if not games:
        return 1.0, 1.0, 0.0

    # Train/val split for α selection (Brier is competition metric)
    rng = np.random.RandomState(42)
    idx = np.arange(len(games))
    rng.shuffle(idx)
    n_val = max(1, int(0.2 * len(games)))
    val_idx = set(idx[:n_val])
    train_games = [g for i, g in enumerate(games) if i not in val_idx]
    val_games = [games[i] for i in val_idx]

    best_alpha, best_c, best_intercept, best_brier = 1.0, 1.0, 0.0, np.inf

    for alpha in [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
        debiased = apply_power_debias(odds_lookup, alpha)

        def _build_xy(glist):
            X_list, y_list = [], []
            LOG_DIFF_CLIP = 15.0  # avoid overflow in LogisticRegression gradient
            for season, t1, t2, actual in glist:
                s1 = _get_log_debiased_strength(debiased, season, t1, seed_map)
                s2 = _get_log_debiased_strength(debiased, season, t2, seed_map)
                diff = np.clip(s1 - s2, -LOG_DIFF_CLIP, LOG_DIFF_CLIP)
                X_list.append(diff)
                y_list.append(float(actual))
            return np.array(X_list).reshape(-1, 1), np.array(y_list)

        X_train, y_train = _build_xy(train_games)
        X_val, y_val = _build_xy(val_games)

        lr = LogisticRegression(
            fit_intercept=True,
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
        )
        lr.fit(X_train, y_train)
        c = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])

        probs_val = lr.predict_proba(X_val)[:, 1]
        probs_val = np.clip(probs_val, 1e-15, 1 - 1e-15)
        brier = np.mean((probs_val - y_val) ** 2)
        if brier < best_brier:
            best_brier = brier
            best_alpha = alpha
            best_c = c
            best_intercept = intercept

    # Refit on all games with best α
    debiased = apply_power_debias(odds_lookup, best_alpha)
    LOG_DIFF_CLIP = 15.0
    X_all, y_all = [], []
    for season, t1, t2, actual in games:
        s1 = _get_log_debiased_strength(debiased, season, t1, seed_map)
        s2 = _get_log_debiased_strength(debiased, season, t2, seed_map)
        diff = np.clip(s1 - s2, -LOG_DIFF_CLIP, LOG_DIFF_CLIP)
        X_all.append(diff)
        y_all.append(float(actual))
    X_all = np.array(X_all).reshape(-1, 1)
    y_all = np.array(y_all)
    lr_final = LogisticRegression(fit_intercept=True, C=0.1, max_iter=1000, solver="lbfgs")
    lr_final.fit(X_all, y_all)
    best_c = float(lr_final.coef_[0, 0])
    best_intercept = float(lr_final.intercept_[0])

    return best_alpha, best_c, best_intercept


def fit_market_mapping(
    odds_lookup: dict,
    seed_map: dict,
    games: list,
) -> tuple:
    """Fit α (grid search) and c, intercept (logistic regression). Returns (alpha, c, intercept).
    P(i beats j) = σ(intercept + c·(log p_i* - log p_j*)); select α by Brier on validation."""
    return fit_alpha_and_c(odds_lookup, seed_map, games)


def market_prob_for_matchup(
    odds_lookup: dict,
    season: int,
    team1_id: int,
    team2_id: int,
    c: Optional[float] = None,
    seed_map: Optional[dict] = None,
) -> float:
    """P(team1 wins). If c is None, uses naive ratio. Else uses learned sigmoid(c*(r1-r2))."""
    if c is not None:
        return market_prob_learned(odds_lookup, season, team1_id, team2_id, c, seed_map)
    # Legacy: naive ratio
    imp1 = odds_lookup.get((season, team1_id), None)
    imp2 = odds_lookup.get((season, team2_id), None)
    if imp1 is None or imp2 is None:
        return 0.5
    total = imp1 + imp2
    if total <= 0:
        return 0.5
    return imp1 / total

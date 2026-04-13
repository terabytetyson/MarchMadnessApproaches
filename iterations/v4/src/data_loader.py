"""Load competition data, compute team strength (random effects), and build feature lookups."""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from . import config


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a' -> 1, 16."""
    return int(seed_str[1:3])


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV if it exists, return None otherwise."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_data(data_dir: Optional[str] = None) -> dict:
    """Load all competition data and compute features.

    Returns a dict with DataFrames and lookup dictionaries.
    """
    data_dir = data_dir or config.DATA_DIR
    data_dir = Path(data_dir)

    # Required files
    data = {
        "m_teams": pd.read_csv(data_dir / "MTeams.csv"),
        "w_teams": pd.read_csv(data_dir / "WTeams.csv"),
        "m_regular": pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv"),
        "w_regular": pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv"),
        "m_tourney": pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv"),
        "w_tourney": pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv"),
        "m_seeds": pd.read_csv(data_dir / "MNCAATourneySeeds.csv"),
        "w_seeds": pd.read_csv(data_dir / "WNCAATourneySeeds.csv"),
    }

    # Optional files
    data["sample_sub"] = _safe_read_csv(str(data_dir / "SampleSubmissionStage1.csv"))
    data["m_tourney_det"] = _safe_read_csv(str(data_dir / "MNCAATourneyDetailedResults.csv"))
    data["w_tourney_det"] = _safe_read_csv(str(data_dir / "WNCAATourneyDetailedResults.csv"))
    data["m_team_coaches"] = _safe_read_csv(str(data_dir / "MTeamCoaches.csv"))
    data["m_team_conf"] = _safe_read_csv(str(data_dir / "MTeamConferences.csv"))
    data["w_team_conf"] = _safe_read_csv(str(data_dir / "WTeamConferences.csv"))
    data["conferences"] = _safe_read_csv(str(data_dir / "Conferences.csv"))
    data["cities"] = _safe_read_csv(str(data_dir / "Cities.csv"))
    data["m_game_cities"] = _safe_read_csv(str(data_dir / "MGameCities.csv"))
    data["w_game_cities"] = _safe_read_csv(str(data_dir / "WGameCities.csv"))
    data["m_massey"] = _safe_read_csv(str(data_dir / "MMasseyOrdinals.csv"))
    data["m_regular_det"] = _safe_read_csv(str(data_dir / "MRegularSeasonDetailedResults.csv"))
    data["w_regular_det"] = _safe_read_csv(str(data_dir / "WRegularSeasonDetailedResults.csv"))
    data["cities_latlon"] = _safe_read_csv(str(data_dir / "cities_latlon.csv"))

    # Compute team strength (random effects: margin = strength_A - strength_B)
    team_strength = _compute_team_strength(
        data["m_regular"], data["m_tourney"],
        data["w_regular"], data["w_tourney"],
    )
    data["team_strength"] = team_strength

    # Seed lookup: (season, team_id) -> numeric seed
    seed_map = {}
    for df in [data["m_seeds"], data["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(row["Season"], row["TeamID"])] = _parse_seed(row["Seed"])
    data["seed_map"] = seed_map

    # Team name lookup
    data["m_team_names"] = dict(zip(data["m_teams"]["TeamID"], data["m_teams"]["TeamName"]))
    data["w_team_names"] = dict(zip(data["w_teams"]["TeamID"], data["w_teams"]["TeamName"]))

    # Win/loss records: (season, team_id) -> (wins, losses)
    data["m_records"] = _compute_records(data["m_regular"], data["m_tourney"])
    data["w_records"] = _compute_records(data["w_regular"], data["w_tourney"])

    # Last 5 games win ratio: (season, team_id) -> wins_in_last_5 / 5
    data["m_last5"] = _compute_last5(data["m_regular"])
    data["w_last5"] = _compute_last5(data["w_regular"])

    # Coach lookup: (season, team_id) -> coach_name (men only)
    data["coach_map"] = _build_coach_map(data["m_team_coaches"])

    # Conference lookup: (season, team_id) -> conference description
    data["conf_map"] = _build_conf_map(
        data["m_team_conf"], data["w_team_conf"], data["conferences"]
    )
    # Conference abbrev: (season, team_id) -> ConfAbbrev
    data["conf_abbrev"] = _build_conf_abbrev(data["m_team_conf"], data["w_team_conf"])

    # Historical seed-vs-seed stats: (seed1, seed2) -> (win_rate, avg_margin)
    data["seed_vs_seed"] = _compute_seed_vs_seed(
        data["m_tourney"], data["m_tourney_det"],
        data["w_tourney"], data["w_tourney_det"],
        data["seed_map"],
    )

    # Game location: (season, daynum, wteam, lteam) -> (city, state)
    data["game_location"], data["game_city_id"], data["all_games_city"] = _build_game_location(
        data["m_game_cities"], data["w_game_cities"], data["cities"]
    )

    # Massey rank: (season, team_id) -> ordinal rank (use RankingDayNum=133)
    data["massey_rank"] = _build_massey_rank(data["m_massey"])

    # Efficiency stats: AdjO, AdjD, ORtg, DRtg, OSOS, DSOS (from detailed results)
    data["m_eff"], data["w_eff"] = _compute_efficiency_stats(
        data["m_regular_det"], data["w_regular_det"],
        data["team_strength"],
    )

    # Paris-style box score stats (FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF)
    data["m_box"], data["w_box"] = _compute_box_stats(
        data["m_regular_det"], data["w_regular_det"],
    )

    # Conference strength: (season, conf_abbrev) -> mean team strength
    data["conf_strength"] = _compute_conf_strength(
        data["m_team_conf"], data["w_team_conf"],
        data["team_strength"],
    )

    # Team last-game city: (season, team_id, daynum) -> (lat, lon) of city where they played previous game
    data["team_last_game_coords"] = _build_team_last_game_coords(
        data["m_regular"], data["w_regular"],
        data["m_tourney"], data["w_tourney"],
        data["all_games_city"], data["cities_latlon"],
    )

    return data


def _compute_team_strength(m_regular, m_tourney, w_regular, w_tourney):
    """Compute team strength via random effects: margin = strength_A - strength_B.

    Fits ridge regression on design matrix: each game row has +1 for winner, -1 for loser.
    Regularization (ridge) = Gaussian prior N(0, sigma^2) on team strength.
    Returns (season, team_id) -> strength.
    """
    result = {}

    def _fit_strength(regular_df, tourney_df):
        # Fit on regular season only (no tournament leakage for prediction)
        all_games = regular_df.copy()
        all_games["Margin"] = all_games["WScore"] - all_games["LScore"]

        # Build (season, team_id) index (include tourney teams for lookup)
        team_seasons = set()
        for _, row in regular_df.iterrows():
            team_seasons.add((row["Season"], row["WTeamID"]))
            team_seasons.add((row["Season"], row["LTeamID"]))
        for _, row in tourney_df.iterrows():
            team_seasons.add((row["Season"], row["WTeamID"]))
            team_seasons.add((row["Season"], row["LTeamID"]))
        team_seasons = sorted(team_seasons)
        idx = {ts: i for i, ts in enumerate(team_seasons)}

        # Design matrix: row i has +1 at winner col, -1 at loser col
        n_games = len(all_games)
        n_teams = len(team_seasons)
        rows, cols, vals = [], [], []
        for i, row in all_games.iterrows():
            season, w_id, l_id = row["Season"], row["WTeamID"], row["LTeamID"]
            w_col = idx.get((season, w_id))
            l_col = idx.get((season, l_id))
            if w_col is not None and l_col is not None:
                rows.extend([i, i])
                cols.extend([w_col, l_col])
                vals.extend([1.0, -1.0])

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_games, n_teams))
        y = all_games["Margin"].values

        # Ridge = random effects: strength ~ N(0, 1/alpha)
        model = Ridge(alpha=config.TEAM_STRENGTH_ALPHA, fit_intercept=False)
        model.fit(X, y)
        strengths = model.coef_

        for (season, team_id), i in idx.items():
            result[(season, team_id)] = float(strengths[i])

    _fit_strength(m_regular, m_tourney)
    _fit_strength(w_regular, w_tourney)
    return result


def _compute_records(regular_df, tourney_df):
    """(season, team_id) -> (wins, losses)."""
    records = {}
    all_games = pd.concat([regular_df, tourney_df])
    for _, row in all_games.iterrows():
        season = row["Season"]
        w_id, l_id = row["WTeamID"], row["LTeamID"]
        for tid in [w_id, l_id]:
            key = (season, tid)
            if key not in records:
                records[key] = [0, 0]
            if tid == w_id:
                records[key][0] += 1
            else:
                records[key][1] += 1
    return records


def _compute_last5(regular_df):
    """(season, team_id) -> wins_in_last_5 / 5 (ratio)."""
    last5 = {}
    for season in regular_df["Season"].unique():
        season_games = regular_df[regular_df["Season"] == season].sort_values("DayNum", ascending=False)
        team_games = {}
        for _, row in season_games.iterrows():
            w_id, l_id = row["WTeamID"], row["LTeamID"]
            for tid in [w_id, l_id]:
                if tid not in team_games:
                    team_games[tid] = []
                if len(team_games[tid]) < 5:
                    team_games[tid].append(1 if tid == w_id else 0)
        for tid, outcomes in team_games.items():
            last5[(season, tid)] = sum(outcomes) / 5.0 if len(outcomes) == 5 else sum(outcomes) / max(len(outcomes), 1)
    return last5


def _build_coach_map(team_coaches):
    """(season, team_id) -> formatted coach name. Men only."""
    if team_coaches is None:
        return {}
    coach_map = {}
    for _, row in team_coaches.iterrows():
        key = (row["Season"], row["TeamID"])
        name = str(row["CoachName"]).replace("_", " ").title()
        coach_map[key] = name
    return coach_map


def _build_conf_map(m_conf, w_conf, conferences):
    """(season, team_id) -> conference description."""
    conf_map = {}
    conf_desc = {}
    if conferences is not None:
        conf_desc = dict(zip(conferences["ConfAbbrev"], conferences["Description"]))

    for df in [m_conf, w_conf]:
        if df is None:
            continue
        for _, row in df.iterrows():
            key = (row["Season"], row["TeamID"])
            abbrev = row["ConfAbbrev"]
            conf_map[key] = conf_desc.get(abbrev, abbrev)
    return conf_map


def _build_conf_abbrev(m_conf, w_conf):
    """(season, team_id) -> ConfAbbrev."""
    out = {}
    for df in [m_conf, w_conf]:
        if df is None:
            continue
        for _, row in df.iterrows():
            out[(row["Season"], row["TeamID"])] = row["ConfAbbrev"]
    return out


def _compute_seed_vs_seed(m_tourney, m_tourney_det, w_tourney, w_tourney_det, seed_map):
    """(seed1, seed2) -> (win_rate_lower_seed, avg_margin)."""
    stats = {}
    for tourney_df, det_df in [(m_tourney, m_tourney_det), (w_tourney, w_tourney_det)]:
        for _, row in tourney_df.iterrows():
            season = row["Season"]
            w_id, l_id = row["WTeamID"], row["LTeamID"]
            s1 = seed_map.get((season, w_id), 8)
            s2 = seed_map.get((season, l_id), 8)
            seed_lo, seed_hi = min(s1, s2), max(s1, s2)
            key = (seed_lo, seed_hi)
            if key not in stats:
                stats[key] = {"wins_lo": 0, "total": 0, "margins": []}
            stats[key]["total"] += 1
            if seed_lo == s1:  # lower seed won
                stats[key]["wins_lo"] += 1
                margin = row["WScore"] - row["LScore"]
            else:
                margin = row["LScore"] - row["WScore"]
            if det_df is not None:
                det_row = det_df[(det_df["Season"] == season) & (det_df["WTeamID"] == w_id) & (det_df["LTeamID"] == l_id)]
                if not det_row.empty:
                    margin = det_row.iloc[0]["WScore"] - det_row.iloc[0]["LScore"]
            stats[key]["margins"].append(margin)

    result = {}
    for key, v in stats.items():
        win_rate = v["wins_lo"] / v["total"] if v["total"] > 0 else 0.5
        avg_margin = sum(v["margins"]) / len(v["margins"]) if v["margins"] else 0
        result[key] = (win_rate, avg_margin)
    return result


def _build_game_location(m_game_cities, w_game_cities, cities):
    """(season, daynum, wteam, lteam) -> (city, state) and CityID."""
    loc = {}
    city_id_map = {}
    all_games_city = {}  # all games (Regular + NCAA) for last-game lookup
    city_lookup = {}
    if cities is not None:
        city_lookup = dict(zip(cities["CityID"], zip(cities["City"], cities["State"])))

    for df in [m_game_cities, w_game_cities]:
        if df is None:
            continue
        for _, row in df.iterrows():
            key = (row["Season"], row["DayNum"], row["WTeamID"], row["LTeamID"])
            key_alt = (row["Season"], row["DayNum"], row["LTeamID"], row["WTeamID"])
            cid = row["CityID"]
            all_games_city[key] = all_games_city[key_alt] = cid
            if row.get("CRType") == "NCAA":
                city_id_map[key] = city_id_map[key_alt] = cid
                loc[key] = loc[key_alt] = city_lookup.get(cid, ("Unknown", ""))

    return loc, city_id_map, all_games_city


def _build_massey_rank(m_massey):
    """(season, team_id) -> ordinal rank. Use RankingDayNum=133 for pre-tournament."""
    if m_massey is None:
        return {}
    rank = {}
    pre_tourney = m_massey[m_massey["RankingDayNum"] == 133]
    for _, row in pre_tourney.iterrows():
        key = (row["Season"], row["TeamID"])
        rank[key] = row["OrdinalRank"]
    return rank


def _possessions_from_row(row, prefix):
    """Possessions = FGA - OR + TO + 0.475*FTA."""
    fga = row.get(f"{prefix}FGA", 0) or 0
    orb = row.get(f"{prefix}OR", 0) or 0
    to = row.get(f"{prefix}TO", 0) or 0
    fta = row.get(f"{prefix}FTA", 0) or 0
    return fga - orb + to + 0.475 * fta


def _compute_efficiency_stats(m_det, w_det, team_strength):
    """(season, team_id) -> {ortg, drtg, adjo, adjd, osos, dsos}.

    ORtg/DRtg = points per 100 poss (raw). AdjO/AdjD = opponent-adjusted.
    OSOS = avg opponent DRtg when we scored. DSOS = avg opponent ORtg when we defended.
    """
    def _run(det_df):
        if det_df is None or len(det_df) == 0:
            return {}
        # Per-game stats
        games = []
        for _, row in det_df.iterrows():
            season, w_id, l_id = row["Season"], row["WTeamID"], row["LTeamID"]
            w_pts = row["WScore"]
            l_pts = row["LScore"]
            w_poss = _possessions_from_row(row, "W")
            l_poss = _possessions_from_row(row, "L")
            poss = (w_poss + l_poss) / 2 if (w_poss + l_poss) > 0 else 70
            games.append({
                "season": season, "w_id": w_id, "l_id": l_id,
                "w_pts": w_pts, "l_pts": l_pts, "poss": poss,
            })
        gdf = pd.DataFrame(games)

        # Aggregate per (season, team_id)
        team_stats = {}
        for (season, team_id), grp in gdf.groupby(["season", "w_id"]):
            pts = grp["w_pts"].sum()
            opp_pts = grp["l_pts"].sum()
            poss = grp["poss"].sum()
            n = len(grp)
            if (season, team_id) not in team_stats:
                team_stats[(season, team_id)] = {"pts": 0, "opp_pts": 0, "poss": 0, "opponents": []}
            team_stats[(season, team_id)]["pts"] += pts
            team_stats[(season, team_id)]["opp_pts"] += opp_pts
            team_stats[(season, team_id)]["poss"] += poss
            team_stats[(season, team_id)]["opponents"].extend(grp["l_id"].tolist())
        for (season, team_id), grp in gdf.groupby(["season", "l_id"]):
            pts = grp["l_pts"].sum()
            opp_pts = grp["w_pts"].sum()
            poss = grp["poss"].sum()
            if (season, team_id) not in team_stats:
                team_stats[(season, team_id)] = {"pts": 0, "opp_pts": 0, "poss": 0, "opponents": []}
            team_stats[(season, team_id)]["pts"] += pts
            team_stats[(season, team_id)]["opp_pts"] += opp_pts
            team_stats[(season, team_id)]["poss"] += poss
            team_stats[(season, team_id)]["opponents"].extend(grp["w_id"].tolist())

        # Raw ORtg, DRtg
        result = {}
        for (season, team_id), s in team_stats.items():
            poss = s["poss"] or 100
            ortg = 100 * s["pts"] / poss
            drtg = 100 * s["opp_pts"] / poss
            result[(season, team_id)] = {"ortg": ortg, "drtg": drtg, "opponents": s["opponents"]}

        # Opponent adjustment (iterative, 5 iterations)
        for _ in range(5):
            updates = {}
            for (season, team_id), s in result.items():
                opps = s["opponents"]
                if not opps:
                    updates[(season, team_id)] = (s["ortg"], s["drtg"])
                    continue
                avg_opp_drtg = np.mean([result.get((season, o), {}).get("drtg", 100) for o in opps]) or 100
                avg_opp_ortg = np.mean([result.get((season, o), {}).get("ortg", 100) for o in opps]) or 100
                adjo = s["ortg"] * (100 / avg_opp_drtg) if avg_opp_drtg else s["ortg"]
                adjd = s["drtg"] * (avg_opp_ortg / 100) if avg_opp_ortg else s["drtg"]
                updates[(season, team_id)] = (adjo, adjd)
            for k, (o, d) in updates.items():
                result[k]["ortg"], result[k]["drtg"] = o, d

        # OSOS, DSOS (use final ortg/drtg after adjustment)
        for (season, team_id), s in result.items():
            opps = s["opponents"]
            if opps:
                s["osos"] = np.mean([result.get((season, o), {}).get("drtg", 100) for o in opps])
                s["dsos"] = np.mean([result.get((season, o), {}).get("ortg", 100) for o in opps])
            else:
                s["osos"] = 100
                s["dsos"] = 100
            s["adjo"] = s["ortg"]
            s["adjd"] = s["drtg"]

        return result

    m_eff = _run(m_det)
    w_eff = _run(w_det)
    return m_eff, w_eff


# Paris-style box score columns (from winning solutions)
_BOX_COLS = ["Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def _compute_box_stats(m_det, w_det):
    """(season, team_id) -> {avg_Score, avg_FGM, ..., avg_opponent_Score, ...}.

    Paris-style: season averages of own performance + opponent performance when playing against this team.
    """
    def _run(det_df):
        if det_df is None or len(det_df) == 0:
            return {}
        # Overtime adjustment: normalize stats by (40 + 5*NumOT)/40
        det_df = det_df.copy()
        if "NumOT" in det_df.columns:
            det_df["adjot"] = (40 + 5 * det_df["NumOT"].fillna(0)) / 40
        else:
            det_df["adjot"] = 1.0
        result = {}
        for _, row in det_df.iterrows():
            season = row["Season"]
            for (w_id, l_id, w_prefix, l_prefix) in [
                (row["WTeamID"], row["LTeamID"], "W", "L"),
                (row["LTeamID"], row["WTeamID"], "L", "W"),
            ]:
                key = (season, w_id)
                if key not in result:
                    result[key] = {c: [] for c in _BOX_COLS}
                    for c in _BOX_COLS:
                        result[key]["opp_" + c] = []
                adjot = row["adjot"]
                for c in _BOX_COLS:
                    w_col = f"{w_prefix}{c}"
                    l_col = f"{l_prefix}{c}"
                    if w_col in row.index and pd.notna(row.get(w_col)):
                        result[key][c].append(row[w_col] / adjot)
                    if l_col in row.index and pd.notna(row.get(l_col)):
                        result[key]["opp_" + c].append(row[l_col] / adjot)

        # Aggregate to means + derive style stats
        out = {}
        for key, d in result.items():
            out[key] = {}
            for c in _BOX_COLS:
                vals = d.get(c, [])
                opp_vals = d.get("opp_" + c, [])
                out[key][c] = np.mean(vals) if vals else 0.0
                out[key]["opp_" + c] = np.mean(opp_vals) if opp_vals else 0.0
            # Style features (how they play, not strength)
            b = out[key]
            fga, fga3, fgm, fta, ast, orb, dr, to, stl, blk, pf = (
                b.get("FGA", 1), b.get("FGA3", 0), b.get("FGM", 0), b.get("FTA", 0),
                b.get("Ast", 0), b.get("OR", 0), b.get("DR", 0), b.get("TO", 0),
                b.get("Stl", 0), b.get("Blk", 0), b.get("PF", 0),
            )
            poss = max(1, fga - orb + to + 0.475 * fta)
            out[key]["style_3par"] = fga3 / fga if fga > 0 else 0.33
            out[key]["style_ftar"] = fta / fga if fga > 0 else 0.25
            out[key]["style_ast_fgm"] = ast / fgm if fgm > 0 else 0.5
            out[key]["style_orb_share"] = orb / (orb + dr + 1e-6)
            out[key]["style_tov_pct"] = to / poss
            out[key]["style_blk_rate"] = blk / poss
            out[key]["style_stl_rate"] = stl / poss
            out[key]["style_pf_rate"] = pf / poss
        return out

    m_box = _run(m_det)
    w_box = _run(w_det)
    return m_box, w_box


def _compute_conf_strength(m_conf, w_conf, team_strength):
    """(season, conf_abbrev) -> mean team strength in that conference."""
    strength_by_conf = {}
    for df in [m_conf, w_conf]:
        if df is None:
            continue
        for _, row in df.iterrows():
            key = (row["Season"], row["ConfAbbrev"])
            tid = row["TeamID"]
            s = team_strength.get((row["Season"], tid), 0)
            if key not in strength_by_conf:
                strength_by_conf[key] = []
            strength_by_conf[key].append(s)
    return {k: np.mean(v) if v else 0 for k, v in strength_by_conf.items()}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _build_team_last_game_coords(m_regular, w_regular, m_tourney, w_tourney, all_games_city, cities_latlon):
    """(season, team_id, daynum) -> (lat, lon) of city where they played their previous game."""
    if all_games_city is None or cities_latlon is None or (hasattr(cities_latlon, 'empty') and cities_latlon.empty):
        return {}
    city_latlon = {}
    for _, row in cities_latlon.iterrows():
        if pd.notna(row.get("Lat")) and pd.notna(row.get("Lon")):
            city_latlon[int(row["CityID"])] = (float(row["Lat"]), float(row["Lon"]))

    # Build (season, team_id) -> [(daynum, city_id), ...] sorted by daynum
    team_games = {}
    for df in [m_regular, w_regular, m_tourney, w_tourney]:
        if df is None:
            continue
        for _, row in df.iterrows():
            season, day, w_id, l_id = row["Season"], row["DayNum"], row["WTeamID"], row["LTeamID"]
            cid = all_games_city.get((season, day, w_id, l_id)) or all_games_city.get((season, day, l_id, w_id))
            if cid is not None:
                for tid in [w_id, l_id]:
                    key = (season, tid)
                    if key not in team_games:
                        team_games[key] = []
                    team_games[key].append((day, cid))

    # Sort by daynum, dedupe by day (take last game per day if multiple)
    for key in team_games:
        by_day = {}
        for d, c in team_games[key]:
            by_day[d] = c
        team_games[key] = sorted(by_day.items())

    # (season, team_id, daynum) -> coords of last game before daynum
    result = {}
    for (season, team_id), games in team_games.items():
        for i, (daynum, cid) in enumerate(games):
            if i == 0:
                continue
            prev_day, prev_cid = games[i - 1]
            if prev_cid in city_latlon:
                result[(season, team_id, daynum)] = city_latlon[prev_cid]
    return result


def get_team_info(data: dict, team_id: int, season: int, gender: str) -> dict:
    """Build team info dict for text_builder."""
    is_men = 1000 <= team_id < 2000
    prefix = "m" if is_men else "w"
    team_names = data["m_team_names"] if is_men else data["w_team_names"]
    records = data["m_records"] if is_men else data["w_records"]
    last5 = data["m_last5"] if is_men else data["w_last5"]

    rec = records.get((season, team_id), [0, 0])
    wins, losses = rec[0], rec[1]
    l5 = last5.get((season, team_id), 0.5)

    strength_val = data["team_strength"].get((season, team_id), 0.0)
    seed_val = data["seed_map"].get((season, team_id), 8)
    coach = data["coach_map"].get((season, team_id), "Unknown") if is_men else "Unknown"
    conf = data["conf_map"].get((season, team_id), "Unknown")
    massey = data["massey_rank"].get((season, team_id))

    eff = data["m_eff"] if is_men else data["w_eff"]
    e = eff.get((season, team_id), {})
    ortg = e.get("ortg", 100)
    drtg = e.get("drtg", 100)
    adjo = e.get("adjo", ortg)
    adjd = e.get("adjd", drtg)
    osos = e.get("osos", 100)
    dsos = e.get("dsos", 100)

    conf_abbrev = data.get("conf_abbrev", {}).get((season, team_id), "")
    conf_str = data.get("conf_strength", {}).get((season, conf_abbrev), 0.0)

    # Paris-style box stats (FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF)
    box = (data["m_box"] if is_men else data["w_box"]).get((season, team_id), {})
    _style_keys = ["style_3par", "style_ftar", "style_ast_fgm", "style_orb_share", "style_tov_pct", "style_blk_rate", "style_stl_rate", "style_pf_rate"]
    _style_defaults = [0.33, 0.25, 0.5, 0.33, 0.15, 0.03, 0.08, 0.2]
    style = {k: box.get(k, d) for k, d in zip(_style_keys, _style_defaults)}

    return {
        "team_id": team_id,
        "team_name": team_names.get(team_id, str(team_id)),
        "seed": seed_val,
        "wins": wins,
        "losses": losses,
        "record_str": f"{wins}-{losses}",
        "strength": strength_val,
        "last5_ratio": l5,
        "last5_wins": int(round(l5 * 5)),
        "last5_total": 5,
        "coach": coach,
        "conference": conf,
        "massey_rank": massey,
        "ortg": ortg,
        "drtg": drtg,
        "adjo": adjo,
        "adjd": adjd,
        "osos": osos,
        "dsos": dsos,
        "conf_strength": conf_str,
        "box": box,
        "style": style,
    }

"""Build text for v8: Paris features (box + style) + market odds. Natural-language stat sentences."""

from typing import Optional

# Paris: box stats + style stats
_PARIS_BOX_ORDER = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
_PARIS_STYLE_ORDER = ["style_3par", "style_ftar", "style_ast_fgm", "style_orb_share", "style_tov_pct", "style_blk_rate", "style_stl_rate", "style_pf_rate"]
_MATCHUP_STAT_ORDER = _PARIS_BOX_ORDER + _PARIS_STYLE_ORDER

_BIG_ADV = 0.70
_SLIGHT_ADV = 0.55
_EVEN_LO, _EVEN_HI = 0.45, 0.55
_SLIGHT_DISADV = 0.30


def _ratio_stat(v1: float, v2: float, eps: float = 1e-6) -> float:
    denom = v1 + v2 + eps
    return max(0, min(1, v1 / denom))


def _value_to_bucket(v: float) -> str:
    if v >= _BIG_ADV:
        return "big_adv"
    if v >= _SLIGHT_ADV:
        return "slight_adv"
    if _EVEN_LO <= v < _EVEN_HI:
        return "even"
    if v >= _SLIGHT_DISADV:
        return "slight_disadv"
    return "big_disadv"


_STAT_DISPLAY = {
    "FGM": "field goals made",
    "FGA": "field goal attempts",
    "FGM3": "three-pointers made",
    "FGA3": "three-point attempts",
    "FTM": "free throws made",
    "FTA": "free throw attempts",
    "OR": "offensive rebounding",
    "DR": "defensive rebounding",
    "Ast": "assists",
    "TO": "turnovers",
    "Stl": "steals",
    "Blk": "blocks",
    "PF": "personal fouls",
    "style_3par": "three-point attempt rate",
    "style_ftar": "free throw attempt rate",
    "style_ast_fgm": "assist rate",
    "style_orb_share": "offensive rebound share",
    "style_tov_pct": "turnover rate",
    "style_blk_rate": "block rate",
    "style_stl_rate": "steal rate",
    "style_pf_rate": "foul rate",
}

_STAT_CONTEXT = {
    k: {
        "big_adv": "holding a commanding edge in this metric",
        "slight_adv": "holding a narrow edge",
        "even": "offering little to no separation between the two teams",
        "slight_disadv": "trailing slightly",
        "big_disadv": "with {n2} holding a commanding edge in this metric",
    }
    for k in _MATCHUP_STAT_ORDER
}


def _stat_to_sentence(stat_name: str, value: float, team1_info: dict, team2_info: dict) -> str:
    n1, n2 = team1_info["team_name"], team2_info["team_name"]
    bucket = _value_to_bucket(value)
    stat_display = _STAT_DISPLAY.get(stat_name, stat_name.lower())
    context = _STAT_CONTEXT.get(stat_name, {}).get(bucket, "offering little to no separation between the two teams")
    context_str = context.format(n1=n1, n2=n2)

    if bucket in ("big_adv", "slight_adv"):
        magnitude = "commanding" if bucket == "big_adv" else "slight"
        return f"{n1} has a {magnitude} {stat_display} advantage of {value:.2f} over {n2}, {context_str}."
    if bucket in ("big_disadv", "slight_disadv"):
        magnitude = "commanding" if bucket == "big_disadv" else "slight"
        return f"{n1} has a {magnitude} {stat_display} disadvantage of {value:.2f} over {n2}, {context_str}."
    return f"{n1} has a negligible {stat_display} difference of {value:.2f} over {n2}, {context_str}."


def _conf_suffix(conf: str) -> str:
    if not conf or conf == "Unknown":
        return "Unknown conference"
    c = conf.strip().lower()
    if c.endswith("conference") or c.endswith("league"):
        return conf.strip()
    return f"{conf.strip()} conference"


def compute_matchup_stats_paris(team1_info: dict, team2_info: dict) -> dict:
    """Compute Paris-style matchup stats: box + style as Team1/(T1+T2) ratio."""
    stats = {}
    eps = 1e-6
    box1 = team1_info.get("box", {})
    box2 = team2_info.get("box", {})

    for col in _PARIS_BOX_ORDER:
        v1 = box1.get(col, 0) or 0
        v2 = box2.get(col, 0) or 0
        stats[col] = _ratio_stat(v1, v2, eps)

    style1 = team1_info.get("style", {})
    style2 = team2_info.get("style", {})
    defaults = {"style_3par": 0.33, "style_ftar": 0.25, "style_ast_fgm": 0.5, "style_orb_share": 0.33,
                "style_tov_pct": 0.15, "style_blk_rate": 0.03, "style_stl_rate": 0.08, "style_pf_rate": 0.2}
    for col in _PARIS_STYLE_ORDER:
        v1 = style1.get(col, defaults.get(col, 0.5)) or defaults.get(col, 0.5)
        v2 = style2.get(col, defaults.get(col, 0.5)) or defaults.get(col, 0.5)
        stats[col] = _ratio_stat(v1, v2, eps)

    return stats


def matchup_to_text(
    team1_info: dict,
    team2_info: dict,
    season: int,
    gender: str,
    location: str = "Neutral site",
    seed_vs_seed: Optional[tuple] = None,
    matchup_stats: Optional[dict] = None,
    market_prob: Optional[float] = None,
) -> str:
    """Build text: matchup, teams, historical, Paris stat sentences, market odds."""
    s1, s2 = team1_info["seed"], team2_info["seed"]
    n1, n2 = team1_info["team_name"], team2_info["team_name"]

    lines = [
        "NCAA Tournament Matchup:",
        f"({s1}) {n1} vs ({s2}) {n2}",
        f"Game played in {location}",
        "",
        f"{n1} has a record of {team1_info['record_str']}, is coached by {team1_info['coach']} and plays in the {_conf_suffix(team1_info['conference'])}.",
        "",
        f"{n2} has a record of {team2_info['record_str']}, is coached by {team2_info['coach']} and plays in the {_conf_suffix(team2_info['conference'])}.",
        "",
    ]

    if seed_vs_seed is not None:
        win_rate, avg_margin = seed_vs_seed
        lower = min(s1, s2)
        lines.append(f"Historically, in a ({s1}) vs ({s2}) matchup, the {lower} seed wins {win_rate * 100:.2f}% of the time by an average margin of {avg_margin:.2f} points.")
        lines.append("")

    stats = matchup_stats if matchup_stats is not None else compute_matchup_stats_paris(team1_info, team2_info)
    for k in _MATCHUP_STAT_ORDER:
        if k in stats:
            lines.append(_stat_to_sentence(k, stats[k], team1_info, team2_info))

    if market_prob is not None:
        pct = market_prob * 100
        lines.append("")
        lines.append(f"Market odds: {n1} has a {pct:.1f}% chance to win according to championship-implied odds.")

    return "\n".join(lines)

"""Load 2026 championship odds and convert to implied probabilities."""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

# Manual mapping: odds JSON name (normalized) -> TeamID for 2026 bracket teams
ODDS_NAME_TO_TEAM_ID = {
    "duke blue devils": 1181,
    "michigan wolverines": 1276,
    "arizona wildcats": 1112,
    "florida gators": 1196,
    "houston cougars": 1222,
    "iowa state cyclones": 1235,
    "illinois fighting illini": 1228,
    "purdue boilermakers": 1345,
    "uconn huskies": 1163,
    "connecticut huskies": 1163,
    "michigan state spartans": 1277,
    "gonzaga bulldogs": 1211,
    "arkansas razorbacks": 1116,
    "kansas jayhawks": 1242,
    "st. john's red storm": 1385,
    "st john's red storm": 1385,
    "virginia cavaliers": 1438,
    "vanderbilt commodores": 1435,
    "wisconsin badgers": 1458,
    "nebraska cornhuskers": 1304,
    "texas tech red raiders": 1403,
    "tennessee volunteers": 1397,
    "louisville cardinals": 1257,
    "ucla bruins": 1417,
    "alabama crimson tide": 1104,
    "ohio state buckeyes": 1326,
    "kentucky wildcats": 1246,
    "saint mary's gaels": 1388,
    "iowa hawkeyes": 1234,
    "north carolina tar heels": 1314,
    "byu cougars": 1140,
    "texas longhorns": 1400,
    "clemson tigers": 1155,
    "miami hurricanes": 1274,
    "villanova wildcats": 1437,
    "georgia bulldogs": 1208,
    "santa clara broncos": 1365,
    "utah state aggies": 1429,
    "missouri tigers": 1281,
    "smu mustangs": 1374,
    "vcu rams": 1433,
    "tcu horned frogs": 1395,
    "texas a&m aggies": 1401,
    "saint louis billikens": 1387,
    "ucf knights": 1416,
    "south florida bulls": 1378,
    "mcneese cowboys": 1270,
    "akron zips": 1103,
    "miami (oh) redhawks": 1275,
    "pennsylvania quakers": 1335,
    "northern iowa panthers": 1320,
    "north dakota state bison": 1295,
    "hofstra pride": 1220,
    "high point panthers": 1219,
    "troy trojans": 1407,
    "hawai'i rainbow warriors": 1218,
    "hawaii rainbow warriors": 1218,
    "california baptist lancers": 1465,
    "howard bison": 1224,
    "long island university sharks": 1254,
    "liu brooklyn": 1254,
    "idaho vandals": 1225,
    "kennesaw state owls": 1244,
    "queens university royals": 1474,
    "lehigh mountain hawks": 1250,
    "furman paladins": 1202,
    "siena saints": 1373,
    "wright state raiders": 1460,
    "tennessee state tigers": 1398,
    "prairie view a&m panthers": 1341,
    "umbc": 1420,
}


def _american_to_prob(odds_str: str) -> float:
    """Convert American odds to implied probability. +360 -> 0.217, -200 -> 0.667."""
    s = str(odds_str).strip()
    if not s or s == "nan":
        return 0.5
    # Fractional: 12-1, 18-1, 60-1
    m = re.match(r"(\d+)-1", s)
    if m:
        a = int(m.group(1))
        return 1.0 / (a + 1) if a > 0 else 0.5
    # American: +360, -135
    m = re.match(r"([+-]?\d+)", s)
    if m:
        val = int(m.group(1))
        if val >= 0:
            return 100 / (val + 100)
        return abs(val) / (abs(val) + 100)
    return 0.5


def _normalize_name(name: str) -> str:
    """Lowercase, strip, collapse spaces."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _build_spellings_to_id(data_dir: Optional[str] = None) -> dict:
    """Build normalized_spelling -> TeamID from MTeamSpellings (men's 1000-1999)."""
    data_dir = Path(data_dir or config.DATA_DIR)
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv")
    result = {}
    for _, row in spellings.iterrows():
        tid = int(row["TeamID"])
        if 1000 <= tid < 2000:
            name = _normalize_name(row["TeamNameSpelling"])
            if name and name not in result:
                result[name] = tid
    return result


def load_championship_odds(
    path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> dict:
    """Load championship odds JSON. Returns team_id -> implied_prob (0-1)."""
    path = Path(path or str(Path(config.DATA_DIR) / "championship_odds_2026.json"))
    if isinstance(path, str):
        path = Path(path)
    data_dir = Path(data_dir or config.DATA_DIR)
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    spellings = _build_spellings_to_id(data_dir)
    result = {}

    for t in data.get("teams", []):
        name = _normalize_name(t.get("name", ""))
        odds_str = t.get("to_win", "")
        prob = _american_to_prob(odds_str)

        # 1. Manual mapping first
        tid = ODDS_NAME_TO_TEAM_ID.get(name)
        if tid is not None:
            result[tid] = prob
            continue

        # 2. Try first word (e.g. "duke" from "duke blue devils")
        first = name.split()[0] if name else ""
        tid = ODDS_NAME_TO_TEAM_ID.get(first) or spellings.get(first)
        if tid is not None:
            result[tid] = prob
            continue

        # 3. Spellings lookup by full name or key parts
        tid = spellings.get(name)
        if tid is not None:
            result[tid] = prob
            continue

        # 4. Partial match: spelling in name
        for spell, sid in spellings.items():
            if len(spell) >= 4 and (spell in name or name.startswith(spell)):
                result[sid] = prob
                break

    return result

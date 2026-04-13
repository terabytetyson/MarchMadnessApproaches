"""Paris features: box + style as Team1/(T1+T2) ratio. From raddar/paris-madness notebook."""

import numpy as np

# Paris: box stats + style stats (same as raddar notebook)
PARIS_BOX_ORDER = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]
PARIS_STYLE_ORDER = ["style_3par", "style_ftar", "style_ast_fgm", "style_orb_share", "style_tov_pct", "style_blk_rate", "style_stl_rate", "style_pf_rate"]
FEATURE_ORDER = PARIS_BOX_ORDER + PARIS_STYLE_ORDER + ["market_prob"]


def _ratio_stat(v1: float, v2: float, eps: float = 1e-6) -> float:
    denom = v1 + v2 + eps
    return max(0, min(1, v1 / denom))


def get_paris_features(
    team1_info: dict,
    team2_info: dict,
    market_prob: float = 0.5,
) -> np.ndarray:
    """Build Paris feature vector: box + style ratios + market_prob."""
    eps = 1e-6
    box1 = team1_info.get("box", {})
    box2 = team2_info.get("box", {})
    style1 = team1_info.get("style", {})
    style2 = team2_info.get("style", {})
    defaults = {"style_3par": 0.33, "style_ftar": 0.25, "style_ast_fgm": 0.5, "style_orb_share": 0.33,
                "style_tov_pct": 0.15, "style_blk_rate": 0.03, "style_stl_rate": 0.08, "style_pf_rate": 0.2}

    feats = []
    for col in PARIS_BOX_ORDER:
        v1 = box1.get(col, 0) or 0
        v2 = box2.get(col, 0) or 0
        feats.append(_ratio_stat(v1, v2, eps))
    for col in PARIS_STYLE_ORDER:
        v1 = style1.get(col, defaults.get(col, 0.5)) or defaults.get(col, 0.5)
        v2 = style2.get(col, defaults.get(col, 0.5)) or defaults.get(col, 0.5)
        feats.append(_ratio_stat(v1, v2, eps))
    feats.append(max(0, min(1, market_prob)))
    return np.array(feats, dtype=np.float32)

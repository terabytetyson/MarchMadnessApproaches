"""Evaluate and generate submission for v9: Paris + market, ExtraTrees. Men's + women's."""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from deps import load_data, get_team_info, load_odds, apply_power_debias, market_prob_learned

from src import config
from src.features import get_paris_features


def _load_model_and_market(model_path: str):
    model_path = Path(model_path)
    with open(model_path / "paris_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_path / "market_params.pkl", "rb") as f:
        d = pickle.load(f)
    return model, d.get("alpha", 1.0), d.get("c", 1.0), d.get("intercept", 0.0)


def evaluate(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    model_path: Optional[str] = None,
    test_years: Optional[list] = None,
    clip_extreme: bool = False,
) -> dict:
    """Evaluate v9 on men's and women's tournament games. Reports Brier for each and cumulative."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    model_path = model_path or os.path.join(config.MODEL_DIR, "paris-v9")
    test_years = set(test_years or config.TEST_YEARS)

    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir) if odds_path and os.path.exists(odds_path) else {}
    model, alpha, c, intercept = _load_model_and_market(model_path)
    debiased_lookup = apply_power_debias(odds_lookup, alpha) if odds_lookup else {}
    seed_map = data.get("seed_map", {})

    women_start = getattr(config, "WOMEN_DETAILED_START", 2010)

    def _run_games(tourney_df, gender: str):
        subset = tourney_df[tourney_df["Season"].isin(test_years)]
        if gender == "women":
            subset = subset[subset["Season"] >= women_start]
        X_list, labels = [], []
        for _, row in subset.iterrows():
            season = int(row["Season"])
            w_id, l_id = row["WTeamID"], row["LTeamID"]
            team1_id, team2_id = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == team1_id else 0

            team1_info = get_team_info(data, team1_id, season, gender)
            team2_info = get_team_info(data, team2_id, season, gender)
            if gender == "men" and debiased_lookup:
                market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)
            else:
                market_prob = 0.5

            feats = get_paris_features(team1_info, team2_info, market_prob=market_prob)
            X_list.append(feats)
            labels.append(label)
        return np.vstack(X_list) if X_list else np.zeros((0, 22)), np.array(labels) if labels else np.array([])

    X_men, labels_men = _run_games(data["m_tourney"], "men")
    X_women, labels_women = _run_games(data["w_tourney"], "women")

    results = {}
    all_preds, all_labels = [], []

    if len(labels_men) > 0:
        probs_men = model.predict_proba(X_men)[:, 1]
        preds_men = np.clip(probs_men, 0.05, 0.95)
        if clip_extreme:
            preds_men = np.where(preds_men > 0.95, 1.0, np.where(preds_men < 0.05, 0.0, preds_men))
        brier_men = np.mean((preds_men - labels_men) ** 2)
        acc_men = np.mean((preds_men > 0.5) == labels_men)
        results["men"] = {"brier": brier_men, "accuracy": acc_men, "n_games": len(labels_men)}
        all_preds.extend(preds_men.tolist())
        all_labels.extend(labels_men.tolist())

    if len(labels_women) > 0:
        probs_women = model.predict_proba(X_women)[:, 1]
        preds_women = np.clip(probs_women, 0.05, 0.95)
        if clip_extreme:
            preds_women = np.where(preds_women > 0.95, 1.0, np.where(preds_women < 0.05, 0.0, preds_women))
        brier_women = np.mean((preds_women - labels_women) ** 2)
        acc_women = np.mean((preds_women > 0.5) == labels_women)
        results["women"] = {"brier": brier_women, "accuracy": acc_women, "n_games": len(labels_women)}
        all_preds.extend(preds_women.tolist())
        all_labels.extend(labels_women.tolist())

    if all_preds:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        brier_cumulative = np.mean((all_preds - all_labels) ** 2)
        acc_cumulative = np.mean((all_preds > 0.5) == all_labels)
        results["cumulative"] = {"brier": brier_cumulative, "accuracy": acc_cumulative, "n_games": len(all_labels)}

    print(f"\n--- V9 Paris + market (ExtraTrees) Evaluation on test years {sorted(test_years)} ---")
    if "men" in results:
        print(f"Men's:   {results['men']['n_games']} games, Brier: {results['men']['brier']:.4f}, Accuracy: {results['men']['accuracy']:.4f}")
    if "women" in results:
        print(f"Women's: {results['women']['n_games']} games, Brier: {results['women']['brier']:.4f}, Accuracy: {results['women']['accuracy']:.4f}")
    if "cumulative" in results:
        print(f"Cumulative: {results['cumulative']['n_games']} games, Brier: {results['cumulative']['brier']:.4f}, Accuracy: {results['cumulative']['accuracy']:.4f}")

    if not results:
        print("No games found in test years.")
        return {"men": None, "women": None, "cumulative": None}

    return results


def generate_submission(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    stage: str = "Stage1",
) -> str:
    """Generate submission.csv. Men's + women's games."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    model_path = model_path or os.path.join(config.MODEL_DIR, "paris-v9")
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "submission.csv")

    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir) if odds_path and os.path.exists(odds_path) else {}
    model, alpha, c, intercept = _load_model_and_market(model_path)
    debiased_lookup = apply_power_debias(odds_lookup, alpha) if odds_lookup else {}
    seed_map = data.get("seed_map", {})

    sample_path = Path(data_dir) / f"SampleSubmission{stage}.csv"
    if not sample_path.exists():
        sample_path = Path(data_dir) / "SampleSubmissionStage1.csv"
    sub = pd.read_csv(sample_path)

    preds = [0.5] * len(sub)
    for pos in tqdm(range(len(sub)), total=len(sub), desc="Generating"):
        row = sub.iloc[pos]
        parts = row["ID"].split("_")
        season, team1_id, team2_id = int(parts[0]), int(parts[1]), int(parts[2])

        is_women = team1_id >= 2000
        gender = "women" if is_women else "men"
        women_start = getattr(config, "WOMEN_DETAILED_START", 2010)
        if is_women and season < women_start:
            preds[pos] = 0.5  # no box data for women before 2010
            continue

        team1_info = get_team_info(data, team1_id, season, gender)
        team2_info = get_team_info(data, team2_id, season, gender)

        if gender == "men" and debiased_lookup:
            market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)
        else:
            market_prob = 0.5

        feats = get_paris_features(team1_info, team2_info, market_prob=market_prob)
        prob = model.predict_proba(feats.reshape(1, -1))[0, 1]
        preds[pos] = float(np.clip(prob, 0.05, 0.95))

    sub["Pred"] = preds
    sub.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    return output_path

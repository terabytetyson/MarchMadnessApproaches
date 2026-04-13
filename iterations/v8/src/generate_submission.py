"""Evaluate and generate submission for v8: Paris text + market odds, DeBERTa. Men's only."""

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from deps import load_data, get_team_info, load_odds, apply_power_debias, market_prob_learned

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.text_builder import matchup_to_text, compute_matchup_stats_paris
from src.train_model import DeBERTaForProbability, _load_model_from_path

INFERENCE_BATCH_SIZE = 64


def _load_model_and_market_params(model_path: str, use_cpu: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = _load_model_from_path(model_path, use_cpu=use_cpu)

    market_path = os.path.join(model_path, "market_params.pkl")
    alpha, c, intercept = 1.0, 1.0, 0.0
    if os.path.exists(market_path):
        with open(market_path, "rb") as f:
            d = pickle.load(f)
        alpha = d.get("alpha", 1.0)
        c = d.get("c", 1.0)
        intercept = d.get("intercept", 0.0)

    return tokenizer, model, alpha, c, intercept


def evaluate(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    model_path: Optional[str] = None,
    test_years: Optional[list] = None,
    use_cpu: bool = False,
    clip_extreme: bool = False,
) -> dict:
    """Evaluate v8 on men's tournament games."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    model_path = model_path or os.path.join(config.MODEL_DIR, "deberta-v8")
    test_years = set(test_years or config.TEST_YEARS)

    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir)
    tokenizer, model, alpha, c, intercept = _load_model_and_market_params(model_path, use_cpu)
    debiased_lookup = apply_power_debias(odds_lookup, alpha)
    seed_map = data.get("seed_map", {})

    tourney_df = data["m_tourney"]
    subset = tourney_df[tourney_df["Season"].isin(test_years)]

    rows = []
    for _, row in subset.iterrows():
        season = int(row["Season"])
        daynum = int(row["DayNum"])
        w_id, l_id = row["WTeamID"], row["LTeamID"]
        team1_id, team2_id = min(w_id, l_id), max(w_id, l_id)
        label = 1 if w_id == team1_id else 0

        team1_info = get_team_info(data, team1_id, season, "men")
        team2_info = get_team_info(data, team2_id, season, "men")
        seed_vs_seed = data["seed_vs_seed"].get((min(team1_info["seed"], team2_info["seed"]), max(team1_info["seed"], team2_info["seed"]))) if data.get("seed_vs_seed") else None
        loc_key = (season, daynum, w_id, l_id)
        loc_alt = (season, daynum, l_id, w_id)
        game_loc = data["game_location"].get(loc_key) or data["game_location"].get(loc_alt)
        location = f"{game_loc[0]}, {game_loc[1]}" if game_loc else "Neutral site"
        matchup_stats = compute_matchup_stats_paris(team1_info, team2_info)
        market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)

        text = matchup_to_text(
            team1_info, team2_info, season, "men",
            location=location,
            seed_vs_seed=seed_vs_seed,
            matchup_stats=matchup_stats,
            market_prob=market_prob,
        )
        rows.append({"text": text, "label": label, "season": season})

    if not rows:
        print("No games found in test years.")
        return {"brier_score": None, "accuracy": None, "n_games": 0}

    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows])

    preds = []
    for i in tqdm(range(0, len(texts), INFERENCE_BATCH_SIZE), desc="Eval inference"):
        batch_texts = texts[i : i + INFERENCE_BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding="max_length")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = model(**inputs)["logits"]
            preds.extend(probs.cpu().numpy().tolist())
    preds = np.clip(np.array(preds), 0.05, 0.95)

    if clip_extreme:
        preds = np.where(preds > 0.95, 1.0, np.where(preds < 0.05, 0.0, preds))

    brier = np.mean((preds - labels) ** 2)
    accuracy = np.mean((preds > 0.5) == labels)

    print(f"\n--- V8 DeBERTa (Paris text + market odds) Evaluation on test years {sorted(test_years)} ---")
    print(f"Games: {len(rows)}, Brier: {brier:.4f}, Accuracy: {accuracy:.4f}")

    return {"brier_score": brier, "accuracy": accuracy, "n_games": len(rows)}


def generate_submission(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    stage: str = "Stage1",
    use_cpu: bool = False,
) -> str:
    """Generate submission.csv. Men's games only."""
    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH
    model_path = model_path or os.path.join(config.MODEL_DIR, "deberta-v8")
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "submission.csv")

    data = load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir)
    tokenizer, model, alpha, c, intercept = _load_model_and_market_params(model_path, use_cpu)
    debiased_lookup = apply_power_debias(odds_lookup, alpha)
    seed_map = data.get("seed_map", {})

    sample_path = Path(data_dir) / f"SampleSubmission{stage}.csv"
    if not sample_path.exists():
        sample_path = Path(data_dir) / "SampleSubmissionStage1.csv"
    sub = pd.read_csv(sample_path)

    texts, indices = [], []
    for pos in tqdm(range(len(sub)), total=len(sub), desc="Building texts"):
        row = sub.iloc[pos]
        parts = row["ID"].split("_")
        season, team1_id, team2_id = int(parts[0]), int(parts[1]), int(parts[2])
        if team1_id >= 2000:
            continue
        indices.append(pos)

        team1_info = get_team_info(data, team1_id, season, "men")
        team2_info = get_team_info(data, team2_id, season, "men")
        seed_vs_seed = data["seed_vs_seed"].get((min(team1_info["seed"], team2_info["seed"]), max(team1_info["seed"], team2_info["seed"]))) if data.get("seed_vs_seed") else None
        matchup_stats = compute_matchup_stats_paris(team1_info, team2_info)
        market_prob = market_prob_learned(debiased_lookup, season, team1_id, team2_id, c, seed_map, intercept=intercept)

        text = matchup_to_text(
            team1_info, team2_info, season, "men",
            location="Neutral site",
            seed_vs_seed=seed_vs_seed,
            matchup_stats=matchup_stats,
            market_prob=market_prob,
        )
        texts.append(text)

    preds = [0.5] * len(sub)
    for i in tqdm(range(0, len(texts), INFERENCE_BATCH_SIZE), desc="Inference"):
        batch_texts = texts[i : i + INFERENCE_BATCH_SIZE]
        batch_indices = indices[i : i + INFERENCE_BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding="max_length")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = model(**inputs)["logits"]
            probs = np.clip(probs.cpu().numpy(), 0.05, 0.95)
        for j, idx in enumerate(batch_indices):
            preds[idx] = float(probs[j])

    sub["Pred"] = preds
    sub.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    return output_path

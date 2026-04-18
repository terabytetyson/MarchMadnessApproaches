"""Unified model comparison framework. Evaluates all approaches (v8, v9, v10, v11 variants)."""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from iterations.v4.deps import load_data as v4_load_data, load_odds
except ImportError:
    pass


class ModelEvaluator:
    """Evaluate different model approaches on consistent test set."""

    def __init__(self, data_dir: str, test_years: List[int] = None):
        self.data_dir = data_dir
        self.test_years = test_years or [2018, 2025]
        self.results = {}

    def _build_test_set(self, gender: str = "men") -> pd.DataFrame:
        """Build test set from tournament results."""
        data = v4_load_data(self.data_dir)
        if gender == "men":
            tourney = data["m_tourney"]
            team_names = data.get("m_team_names", {})
            seed_map = data.get("seed_map", {})
        else:
            tourney = data["w_tourney"]
            team_names = data.get("w_team_names", {})
            seed_map = data.get("seed_map", {})

        subset = tourney[tourney["Season"].isin(self.test_years)]

        rows = []
        for _, row in subset.iterrows():
            season = int(row["Season"])
            w_id = row["WTeamID"]
            l_id = row["LTeamID"]
            w_score = row["WScore"]
            l_score = row["LScore"]

            team1_id = min(w_id, l_id)
            team2_id = max(w_id, l_id)
            label = 1.0 if w_id == team1_id else 0.0

            team1_name = team_names.get(team1_id, f"Team {team1_id}")
            team2_name = team_names.get(team2_id, f"Team {team2_id}")
            seed1 = seed_map.get((season, team1_id), 0)
            seed2 = seed_map.get((season, team2_id), 0)

            rows.append({
                "season": season,
                "team1_id": team1_id,
                "team2_id": team2_id,
                "team1_name": team1_name,
                "team2_name": team2_name,
                "seed1": seed1,
                "seed2": seed2,
                "label": label,
            })

        return pd.DataFrame(rows)

    def evaluate_bert_variant(
        self,
        model_path: str,
        variant_name: str,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Evaluate a BERT variant."""
        try:
            from bert_trainer import BertForProbability
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            bin_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(bin_path):
                ckpt = torch.load(bin_path, map_location="cpu")
                model = BertForProbability(ckpt["model_name"])
                model.load_state_dict(ckpt["state_dict"])
            else:
                print(f"Model checkpoint not found at {model_path}")
                return {}

            model.eval()
            predictions = []

            with torch.no_grad():
                for _, row in test_df.iterrows():
                    text = f"{row['team1_name']} (seed {row['seed1']}) vs {row['team2_name']} (seed {row['seed2']}) in season {row['season']}"
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                    outputs = model(**inputs)
                    pred = outputs["logits"].item()
                    predictions.append(pred)

            predictions = np.array(predictions)
            labels = test_df["label"].values

            metrics = {
                "accuracy": accuracy_score(labels, (predictions > 0.5).astype(int)),
                "auc": roc_auc_score(labels, predictions),
                "brier": brier_score_loss(labels, predictions),
                "logloss": log_loss(labels, predictions),
            }
            return metrics
        except Exception as e:
            print(f"Error evaluating {variant_name}: {e}")
            return {}

    def evaluate_all(self, models_dir: str) -> pd.DataFrame:
        """Evaluate all available models."""
        test_df = self._build_test_set("men")
        print(f"\nTest set: {len(test_df)} games from years {self.test_years}")

        all_results = []

        bert_variants = [
            ("models/bert-base", "BERT-Base"),
            ("models/bert-aggressive", "BERT-Aggressive"),
            ("models/bert-conservative", "BERT-Conservative"),
        ]

        for model_path, variant_name in bert_variants:
            full_path = os.path.join(self.data_dir, "..", model_path)
            if os.path.exists(full_path):
                print(f"\nEvaluating {variant_name}...")
                metrics = self.evaluate_bert_variant(full_path, variant_name, test_df)
                if metrics:
                    all_results.append({
                        "model": variant_name,
                        "type": "BERT",
                        **metrics
                    })

        results_df = pd.DataFrame(all_results)
        return results_df


def generate_comparison_report(results_df: pd.DataFrame, output_path: str = "model_comparison_report.csv"):
    """Generate comparison report."""
    if results_df.empty:
        print("No results to report")
        return

    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--output", default="model_comparison_report.csv", help="Output CSV path")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.data_dir)
    results = evaluator.evaluate_all(args.models_dir)
    generate_comparison_report(results, args.output)

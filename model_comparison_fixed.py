import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

class ModelEvaluator:
    def __init__(self, data_dir, models_dir):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    def load_data(self):
        """Load tournament data"""
        try:
            # Load game results
            games_file = self.data_dir / "MNCAATourneyCompactResults.csv"
            games_df = pd.read_csv(games_file)
            
            # Load seeds
            seeds_file = self.data_dir / "MNCAATourneySeeds.csv"
            seeds_df = pd.read_csv(seeds_file)
            
            # Load team names
            teams_file = self.data_dir / "MTeams.csv"
            teams_df = pd.read_csv(teams_file)
            
            return games_df, seeds_df, teams_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def build_test_set(self, games_df, seeds_df, teams_df):
        """Build test set from tournament games"""
        if any(df is None for df in [games_df, seeds_df, teams_df]):
            print("Warning: Could not load data properly")
            return None, None
        
        try:
            test_data = []
            labels = []
            
            # Filter to 2018 tournament
            test_games = games_df[games_df['Season'] == 2018]
            
            # Create seed mapping
            seeds_2018 = seeds_df[seeds_df['Season'] == 2018]
            seed_map = {}
            for _, row in seeds_2018.iterrows():
                team_id = row['TeamID']
                seed = str(row['Seed']).split('-')[0]  # Extract number from seed
                seed_map[team_id] = seed
            
            # Create team name mapping
            name_map = dict(zip(teams_df['TeamID'], teams_df['TeamName']))
            
            # Build examples
            for _, game in test_games.iterrows():
                team1_id = game['WTeamID']
                team2_id = game['LTeamID']
                
                team1_name = name_map.get(team1_id, f"Team{team1_id}")
                team2_name = name_map.get(team2_id, f"Team{team2_id}")
                seed1 = seed_map.get(team1_id, "0")
                seed2 = seed_map.get(team2_id, "0")
                
                text = f"{team1_name} (seed {seed1}) vs {team2_name} (seed {seed2}) in season 2018"
                test_data.append(text)
                labels.append(1)  # Team 1 won
            
            return test_data, labels
        
        except Exception as e:
            print(f"Error building test set: {e}")
            return None, None
    
    def load_model(self, model_name):
        """Load a single model safely"""
        try:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                print(f"Model path not found: {model_path}")
                return None
            
            model = BertForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def evaluate_model(self, model, test_data):
        """Evaluate a single model"""
        if model is None or not test_data:
            return None
        
        try:
            predictions = []
            with torch.no_grad():
                for text in test_data:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prob = torch.softmax(logits, dim=1)[0][1].item()
                    predictions.append(prob)
            
            accuracy = sum([1 for p in predictions if p > 0.5]) / len(predictions) if predictions else 0
            return {
                "accuracy": accuracy,
                "predictions": predictions,
                "avg_prob": np.mean(predictions)
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
    
    def evaluate_all(self, variants):
        """Evaluate all model variants"""
        # Load data once
        games_df, seeds_df, teams_df = self.load_data()
        test_data, test_labels = self.build_test_set(games_df, seeds_df, teams_df)
        
        if not test_data:
            print("Could not build test set")
            return None
        
        results = []
        
        for variant_name in variants:
            print(f"Evaluating {variant_name}...")
            
            # Load model
            model = self.load_model(variant_name)
            if model is None:
                print(f"Skipping {variant_name} - could not load model")
                continue
            
            # Evaluate
            eval_result = self.evaluate_model(model, test_data)
            if eval_result:
                results.append({
                    "Model": variant_name,
                    "Accuracy": f"{eval_result['accuracy']:.4f}",
                    "Avg_Prob": f"{eval_result['avg_prob']:.4f}"
                })
            
            # Clear model from memory
            del model
            torch.cuda.empty_cache()
        
        return pd.DataFrame(results) if results else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="model_comparison_results.csv")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.data_dir, args.models_dir)
    
    variants = [
        "bert-bert-base",
        "bert-bert-aggressive",
        "bert-bert-conservative"
    ]
    
    results_df = evaluator.evaluate_all(variants)
    
    if results_df is not None:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print(results_df)
    else:
        print("No results to save")

if __name__ == "__main__":
    main()


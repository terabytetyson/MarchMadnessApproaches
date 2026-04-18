"""Generate Kaggle submission from trained BERT model."""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from iterations.v4.deps import load_data as v4_load_data
except ImportError:
    print("Warning: Could not import v4 data loader")


def load_bert_model(model_path: str):
    """Load trained BERT model."""
    from bert_trainer import BertForProbability

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    ckpt = torch.load(bin_path, map_location="cpu")
    model = BertForProbability(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def generate_submission(
    data_dir: str,
    model_path: str,
    output_path: str,
    stage: str = "Stage1",
    use_cpu: bool = False,
):
    """Generate Kaggle submission CSV."""
    print(f"\n{'='*60}")
    print(f"Generating Kaggle submission from {model_path}")
    print(f"{'='*60}\n")

    # Load data and model
    try:
        data = v4_load_data(data_dir)
    except:
        print("Warning: Could not load full data; using minimal setup")
        data = {"m_tourney": None}

    model = load_bert_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not use_cpu:
        if torch.cuda.is_available():
            model = model.cuda()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model = model.to("mps")

    # Load sample submission to know what predictions are expected
    sample_path = os.path.join(data_dir, "SampleSubmissionStage1.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        print(f"Loaded sample submission with {len(sample)} predictions")
    else:
        print("Warning: Sample submission not found")
        return

    seed_map = data.get("seed_map", {})
    team_names = data.get("m_team_names", {})

    predictions = []
    with torch.no_grad():
        for idx, row in sample.iterrows():
            if idx % 100 == 0:
                print(f"Processing prediction {idx}/{len(sample)}...")

            # Extract team IDs from ID column (format: YYYY_T1_T2)
            parts = str(row["ID"]).split("_")
            if len(parts) != 3:
                print(f"Warning: Could not parse ID {row['ID']}")
                predictions.append(0.5)
                continue

            season, team1_id, team2_id = int(parts[0]), int(parts[1]), int(parts[2])

            seed1 = seed_map.get((season, team1_id), 0)
            seed2 = seed_map.get((season, team2_id), 0)
            name1 = team_names.get(team1_id, f"Team {team1_id}")
            name2 = team_names.get(team2_id, f"Team {team2_id}")

            text = f"{name1} (seed {seed1}) vs {name2} (seed {seed2}) in season {season}"

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            if not use_cpu:
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") for k, v in inputs.items()}

            outputs = model(**inputs)
            pred = outputs["logits"].item()
            predictions.append(pred)

    # Create submission
    submission = sample.copy()
    submission["Pred"] = predictions
    submission.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved to {output_path}")
    print(f"  - {len(submission)} predictions")
    print(f"  - Mean predicted prob: {np.mean(predictions):.3f}")
    print(f"  - Min: {np.min(predictions):.3f}, Max: {np.max(predictions):.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from trained BERT")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model", required=True, help="Model variant (base/aggressive/conservative)")
    parser.add_argument("--model-path", default=None, help="Full path to model (overrides --model)")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--stage", default="Stage1", choices=["Stage1", "Stage2"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.model_path:
        model_dir = args.model_path
    else:
        model_dir = os.path.join("models", f"bert-{args.model}")

    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        sys.exit(1)

    generate_submission(
        data_dir=args.data_dir,
        model_path=model_dir,
        output_path=args.output,
        stage=args.stage,
        use_cpu=args.cpu,
    )

"""Unified BERT trainer for all v11 variants. Adapts config at runtime."""

import os
import sys
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    TrainingArguments, Trainer, get_linear_schedule_with_warmup
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root.parent.parent))

try:
    from iterations.v4.deps import (
        load_data as v4_load_data,
        load_odds, apply_power_debias, fit_market_mapping
    )
except ImportError:
    from iterations.v8.deps import (
        load_data as v4_load_data,
        load_odds, apply_power_debias, fit_market_mapping
    )


class BertForProbability(nn.Module):
    """BERT + linear head for probability prediction."""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.config.hidden_size, 1)
        self.model_name = model_name

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        bert_kwargs = {k: v for k, v in kwargs.items() if k in ("token_type_ids",)}
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **bert_kwargs)
        pooler = outputs.last_hidden_state[:, 0]
        pooler = self.dropout(pooler)
        logits = self.regressor(pooler).squeeze(-1)
        probs = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy(probs, labels.float())
        return {"loss": loss, "logits": probs}

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "model_name": self.model_name
        }, os.path.join(path, "pytorch_model.bin"))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    acc = accuracy_score(labels, (predictions > 0.5).astype(int))
    auc = roc_auc_score(labels, predictions)
    brier = brier_score_loss(labels, predictions)
    return {"accuracy": acc, "auc": auc, "brier": brier}


def build_training_data(
    data_dir: str,
    odds_path: str,
    exclude_seasons: Optional[list] = None,
    include_only_seasons: Optional[list] = None,
    debiased_lookup: Optional[dict] = None,
    c: Optional[float] = None,
    intercept: float = 0.0,
    train_start: int = 2003,
    train_end: int = 2024,
) -> pd.DataFrame:
    """Build DataFrame with text for BERT training."""
    data = v4_load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir) if os.path.exists(odds_path) else {}
    seed_map = data.get("seed_map", {})

    if debiased_lookup is None or c is None:
        games_for_fit = []
        tourney = data["m_tourney"]
        subset = tourney[(tourney["Season"] >= train_start) & (tourney["Season"] <= train_end)]
        if exclude_seasons:
            subset = subset[~subset["Season"].isin(exclude_seasons)]
        for _, row in subset.iterrows():
            w_id, l_id = row["WTeamID"], row["LTeamID"]
            team1_id, team2_id = min(w_id, l_id), max(w_id, l_id)
            actual = 1.0 if w_id == team1_id else 0.0
            games_for_fit.append((int(row["Season"]), team1_id, team2_id, actual))
        if odds_lookup:
            alpha, c, intercept = fit_market_mapping(odds_lookup, seed_map, games_for_fit)
            debiased_lookup = apply_power_debias(odds_lookup, alpha)
        else:
            c, intercept = 1.0, 0.0
            debiased_lookup = {}

    exclude = set(exclude_seasons or [])
    include_only = set(include_only_seasons or [])

    tourney_df = data["m_tourney"]
    subset = tourney_df[(tourney_df["Season"] >= train_start) & (tourney_df["Season"] <= train_end)]
    if include_only:
        subset = subset[subset["Season"].isin(include_only)]
    else:
        subset = subset[~subset["Season"].isin(exclude)]

    team_names = data.get("m_team_names", {})
    rows = []
    for _, row in subset.iterrows():
        season = int(row["Season"])
        w_id = row["WTeamID"]
        l_id = row["LTeamID"]
        w_score = row["WScore"]
        l_score = row["LScore"]

        team1_id = min(w_id, l_id)
        team2_id = max(w_id, l_id)
        margin_raw = w_score - l_score
        margin = margin_raw if w_id == team1_id else -margin_raw

        team1_name = team_names.get(team1_id, f"Team {team1_id}")
        team2_name = team_names.get(team2_id, f"Team {team2_id}")

        seed1 = seed_map.get((season, team1_id), 0)
        seed2 = seed_map.get((season, team2_id), 0)

        text = f"{team1_name} (seed {seed1}) vs {team2_name} (seed {seed2}) in season {season}"

        rows.append({
            "text": text,
            "margin": margin,
            "season": season,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "team1_name": team1_name,
            "team2_name": team2_name,
            "winner_id": w_id,
        })

        text_swapped = f"{team2_name} (seed {seed2}) vs {team1_name} (seed {seed1}) in season {season}"
        rows.append({
            "text": text_swapped,
            "margin": -margin,
            "season": season,
            "team1_id": team2_id,
            "team2_id": team1_id,
            "team1_name": team2_name,
            "team2_name": team1_name,
            "winner_id": w_id,
        })

    return pd.DataFrame(rows)


def train_bert(
    config,
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    exclude_seasons: Optional[list] = None,
    use_cpu: bool = False,
) -> Tuple:
    """Train BERT variant."""
    output_dir = output_dir or os.path.join(config.MODEL_DIR, f"bert-{config.MODEL_NAME}")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH

    print(f"\n{'='*60}")
    print(f"Training BERT variant: {config.MODEL_NAME}")
    print(f"{'='*60}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Dropout: {config.BERT_DROPOUT}")
    print(f"{'='*60}\n")

    train_df = build_training_data(
        data_dir=data_dir,
        odds_path=odds_path,
        exclude_seasons=exclude_seasons,
        train_start=config.TRAIN_START,
        train_end=config.TRAIN_END,
    )

    seasons = sorted(train_df["season"].unique())
    n_val = max(1, int(len(seasons) * config.VAL_RATIO))
    val_season_set = set(seasons[-n_val:])
    val_df = train_df[train_df["season"].isin(val_season_set)]
    train_df = train_df[~train_df["season"].isin(val_season_set)]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_df["label"] = (train_df["margin"] > 0).astype(np.float32)
    val_df["label"] = (val_df["margin"] > 0).astype(np.float32)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].rename(columns={"label": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].rename(columns={"label": "labels"}))

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_BASE_MODEL)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    device = "cpu"
    if not use_cpu:
        if torch.cuda.is_available():
            device = f"cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    print(f"Using device: {device}")

    model = BertForProbability(config.BERT_BASE_MODEL, dropout=config.BERT_DROPOUT)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=getattr(config, "WEIGHT_DECAY", 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        use_cpu=use_cpu,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")
    return model, tokenizer

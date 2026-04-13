"""Train DeBERTa for v8: Paris text + market odds, direct probability prediction."""

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from deps import load_data as v4_load_data, load_odds, apply_power_debias, fit_market_mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.build_training_data import build_training_data, print_samples


class DeBERTaForProbability(nn.Module):
    """DeBERTa + linear head for direct probability prediction (sigmoid output)."""

    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.config.hidden_size, 1)
        self.model_name = model_name

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        bert_kwargs = {k: v for k, v in kwargs.items() if k in ("token_type_ids",)}
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **bert_kwargs)
        pooler = outputs.last_hidden_state[:, 0]
        logits = self.regressor(pooler).squeeze(-1)
        probs = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy(probs, labels.float())
        return {"loss": loss, "logits": probs}

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "model_name": self.model_name}, os.path.join(path, "pytorch_model.bin"))


def train(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    exclude_seasons: Optional[list] = None,
    val_seasons: Optional[list] = None,
    use_cpu: bool = False,
    num_epochs: Optional[int] = None,
) -> tuple:
    """Train on Paris text + market odds. Men's only."""
    output_dir = output_dir or os.path.join(config.MODEL_DIR, "deberta-v8")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH

    data = v4_load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir)
    seed_map = data.get("seed_map", {})
    tourney = data["m_tourney"]
    subset = tourney[(tourney["Season"] >= config.TRAIN_START) & (tourney["Season"] <= config.TRAIN_END)]
    if exclude_seasons:
        subset = subset[~subset["Season"].isin(exclude_seasons)]
    games_for_fit = []
    for _, r in subset.iterrows():
        w_id, l_id = r["WTeamID"], r["LTeamID"]
        t1, t2 = min(w_id, l_id), max(w_id, l_id)
        actual = 1.0 if w_id == t1 else 0.0
        games_for_fit.append((int(r["Season"]), t1, t2, actual))
    alpha, c, intercept = fit_market_mapping(odds_lookup, seed_map, games_for_fit)
    with open(os.path.join(output_dir, "market_params.pkl"), "wb") as f:
        pickle.dump({"alpha": alpha, "c": c, "intercept": intercept}, f)
    print(f"Fitted market: α={alpha:.4f}, c={c:.4f}, intercept={intercept:.4f}")

    debiased_lookup = apply_power_debias(odds_lookup, alpha)

    train_df = build_training_data(
        data_dir=data_dir,
        odds_path=odds_path,
        exclude_seasons=exclude_seasons,
        debiased_lookup=debiased_lookup,
        c=c,
        intercept=intercept,
    )
    print_samples(train_df, n=3)

    if val_seasons:
        val_df = build_training_data(
            data_dir=data_dir,
            odds_path=odds_path,
            include_only_seasons=val_seasons,
            debiased_lookup=debiased_lookup,
            c=c,
            intercept=intercept,
        )
        print(f"Train: {len(train_df)} (excl. {exclude_seasons}), Val: {len(val_df)} (seasons {val_seasons})")
    else:
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

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_BASE_MODEL, use_fast=False)

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

    epochs = num_epochs if num_epochs is not None else config.NUM_EPOCHS
    if use_cpu:
        device = "CPU (--cpu)"
    elif torch.cuda.is_available():
        device = f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "GPU (Apple MPS)"
    else:
        device = "CPU (no GPU available)"
    print(f"Using {device} for training", flush=True)

    model = DeBERTaForProbability(config.BERT_BASE_MODEL)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,
        use_cpu=use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def _load_model_from_path(model_path: str, use_cpu: bool = False) -> DeBERTaForProbability:
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(bin_path):
        ckpt = torch.load(bin_path, map_location="cpu")
        model = DeBERTaForProbability(ckpt["model_name"])
        model.load_state_dict(ckpt["state_dict"])
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        model = DeBERTaForProbability(config.BERT_BASE_MODEL)
        model.load_state_dict(load_file(safetensors_path), strict=True)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    if not use_cpu:
        if torch.cuda.is_available():
            model = model.cuda()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model = model.to("mps")
    model.eval()
    return model

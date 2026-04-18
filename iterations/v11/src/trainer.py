"""
trainer.py — Fine-tuning loop for v11 RoBERTa matchup classifier.

Design mirrors v8 (DeBERTa) but removes token_type_ids from every
batch forward call — the one API difference that would cause a silent
bug if ported naively from v8.

Scheduler: linear warmup → linear decay (same as v8).
"""

import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import MatchupDataset
from model import build_model

log = logging.getLogger(__name__)


# ── reproducibility ──────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── collation helper ─────────────────────────────────────────────────────────

def _collate(batch):
    """Stack a list of dataset items into a batched dict."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


# ── main training function ───────────────────────────────────────────────────

def run_training(
    texts: List[str],
    labels: List[int],
    numeric_features: np.ndarray,
    model_name: str = "FacebookAI/roberta-base",
    max_len: int = 128,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    use_cpu: bool = False,
    val_split: float = 0.1,
) -> Tuple:
    """
    Fine-tune RoBERTa on matchup texts.

    Returns (model, tokenizer) on CPU, ready for inference.
    """
    _set_seed(seed)

    device = torch.device("cpu") if use_cpu or not torch.cuda.is_available() \
             else torch.device("cuda")
    log.info("Training device: %s", device)

    # ── tokenizer ────────────────────────────────────────────────────────────
    # AutoTokenizer for roberta-* returns RobertaTokenizerFast
    # It automatically omits token_type_ids for RoBERTa models
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # ── dataset split ─────────────────────────────────────────────────────────
    full_ds = MatchupDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_len=max_len,
        labels=labels,
        numeric_feats=numeric_features,
    )
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    log.info("Train: %d  |  Val: %d", n_train, n_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=_collate, num_workers=0
    )

    # ── model ────────────────────────────────────────────────────────────────
    log.info("Loading RoBERTa model: %s", model_name)
    model = build_model(model_name=model_name, num_numeric=numeric_features.shape[1])
    model.to(device)

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    # Separate weight decay: apply to all params except bias / LayerNorm
    no_decay = {"bias", "LayerNorm.weight"}
    optimizer_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_groups, lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    log.info("Total steps: %d  |  Warmup steps: %d", total_steps, warmup_steps)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # IMPORTANT: no token_type_ids — batch only has input_ids,
            # attention_mask, numeric, labels
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                numeric=batch["numeric"],
                labels=batch["labels"],
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = out["logits"].argmax(dim=-1)
            n_correct += (preds == batch["labels"]).sum().item()
            n_total += len(batch["labels"])

            if (step + 1) % 100 == 0:
                log.info(
                    "  Epoch %d  step %d/%d  loss=%.4f  acc=%.3f",
                    epoch, step + 1, len(train_loader),
                    train_loss / (step + 1), n_correct / n_total
                )

        avg_train_loss = train_loss / len(train_loader)
        train_acc = n_correct / n_total

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numeric=batch["numeric"],
                    labels=batch["labels"],
                )
                val_loss += out["loss"].item()
                preds = out["logits"].argmax(dim=-1)
                v_correct += (preds == batch["labels"]).sum().item()
                v_total += len(batch["labels"])

        avg_val_loss = val_loss / len(val_loader)
        val_acc = v_correct / v_total

        log.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  "
            "val_loss=%.4f  val_acc=%.3f",
            epoch, epochs,
            avg_train_loss, train_acc,
            avg_val_loss, val_acc,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("  ✓ New best val_loss %.4f — checkpoint saved", best_val_loss)

    # ── restore best weights ─────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint (val_loss=%.4f)", best_val_loss)

    model.to(torch.device("cpu"))
    model.eval()
    return model, tokenizer

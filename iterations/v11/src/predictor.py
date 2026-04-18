"""
predictor.py — Inference + submission CSV generation for v11.

Outputs the standard Kaggle format:
    ID,Pred
    2025_1101_1102,0.72
    …
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import pandas as pd

from dataset import MatchupDataset
from model import RobertaMatchupClassifier

log = logging.getLogger(__name__)


def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def predict_proba(
    model: RobertaMatchupClassifier,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    numeric_features: np.ndarray,
    max_len: int = 128,
    batch_size: int = 32,
    use_cpu: bool = False,
) -> np.ndarray:
    """
    Run inference; return probability of label=1 for each matchup.

    Returns
    -------
    np.ndarray of shape (N,) with values in [0, 1].
    """
    device = torch.device("cpu") if use_cpu or not torch.cuda.is_available() \
             else torch.device("cuda")

    model.to(device)
    model.eval()

    ds = MatchupDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_len=max_len,
        labels=None,         # inference — no labels
        numeric_feats=numeric_features,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                numeric=batch["numeric"],
                # no labels at inference time
            )
            # softmax over the 2-class logits → P(label=1)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    model.to(torch.device("cpu"))
    return np.concatenate(all_probs, axis=0)


def generate_submission(
    model: RobertaMatchupClassifier,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    numeric_features: np.ndarray,
    matchup_ids: List[str],
    max_len: int = 128,
    batch_size: int = 32,
    output_dir: str = ".",
    use_cpu: bool = False,
) -> str:
    """
    Generate submission.csv and return its path.
    """
    log.info("Running inference on %d matchups …", len(texts))
    probs = predict_proba(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        numeric_features=numeric_features,
        max_len=max_len,
        batch_size=batch_size,
        use_cpu=use_cpu,
    )

    df = pd.DataFrame({"ID": matchup_ids, "Pred": probs})

    # clip to avoid log-loss explosion at boundaries
    df["Pred"] = df["Pred"].clip(0.025, 0.975)

    out_path = os.path.join(output_dir, "submission_v11.csv")
    df.to_csv(out_path, index=False)
    log.info("Submission written → %s  (%d rows)", out_path, len(df))
    return out_path

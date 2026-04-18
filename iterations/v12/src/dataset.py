"""
dataset.py — PyTorch Dataset for v12 DistilBERT inputs.

Critical DistilBERT API differences from BERT / DeBERTa:
─────────────────────────────────────────────────────────
1. NO token_type_ids
   DistilBERT was distilled from BERT but drops the segment-embedding
   layer entirely.  The model has no token_type_ids attribute and will
   raise an error if you pass them.  AutoTokenizer will NOT return them
   for distilbert-* models; we assert this in __getitem__ during debug.

2. NO pooler_output
   DistilBertModel.forward() returns a BaseModelOutput that has
   last_hidden_state but NO pooler_output field.  (This is unlike
   BERT / RoBERTa which both expose pooler_output via an optional
   pooling layer.)  The classifier in model.py must manually extract
   the [CLS] token:  last_hidden_state[:, 0, :]

3. Same hidden size (768 for distilbert-base-*)
   The numeric fusion head shape is identical to v11.

Supported model ids:
    distilbert-base-uncased     ← default
    distilbert-base-cased
    distilroberta-base          ← RoBERTa-based distil, same rules apply
"""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class MatchupDataset(Dataset):
    """
    Tokenises matchup strings for DistilBERT.

    Parameters
    ----------
    texts          : list of matchup description strings
    tokenizer      : loaded DistilBertTokenizerFast / AutoTokenizer
    max_len        : maximum token sequence length (cap 512)
    labels         : list of int (0/1); None for inference
    numeric_feats  : np.ndarray (N, num_numeric) or None
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
        labels: Optional[List[int]] = None,
        numeric_feats: Optional[np.ndarray] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.numeric_feats = numeric_feats

        if labels is not None and len(labels) != len(texts):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) length mismatch."
            )
        if numeric_feats is not None and len(numeric_feats) != len(texts):
            raise ValueError(
                f"texts ({len(texts)}) and numeric_feats ({len(numeric_feats)}) "
                "length mismatch."
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            # DistilBERT: do NOT request token_type_ids
            # AutoTokenizer omits them automatically for distilbert-*
        )

        # Remove token_type_ids if present (DistilBERT does not use them)
        # The tokenizer might include them due to HF version differences,
        # but we simply ignore them.
        encoding.pop("token_type_ids", None)

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # NO token_type_ids — DistilBERT has no segment embeddings
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.numeric_feats is not None:
            item["numeric"] = torch.tensor(
                self.numeric_feats[idx], dtype=torch.float
            )

        return item

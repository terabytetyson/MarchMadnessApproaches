"""
dataset.py — PyTorch Dataset for v11 RoBERTa inputs.

Critical RoBERTa note from HF docs:
  • RoBERTa does NOT use token_type_ids — all segment ids are zeros
    and the model ignores them.  Do NOT pass token_type_ids to the
    forward call (or pass None).  This differs from DeBERTa-v3 which
    uses them optionally.

  • The tokenizer is backed by HuggingFace's fast tokenizers library
    (Byte-Pair-Encoding, not SentencePiece like DeBERTa-v3).

  • Special tokens: <s> (bos/cls), </s> (eos/sep), <pad> (pad_token_id=1)
"""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class MatchupDataset(Dataset):
    """
    Tokenises matchup text strings for RoBERTa and optionally returns
    a small numeric feature vector (seed gap, Massey gap).

    Parameters
    ----------
    texts          : list of matchup description strings
    tokenizer      : a loaded RobertaTokenizer / AutoTokenizer
    max_len        : maximum token length (RoBERTa cap = 512)
    labels         : list of int (0/1); None for inference
    numeric_feats  : np.ndarray of shape (N, num_numeric); can be None
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
                f"texts ({len(texts)}) and labels ({len(labels)}) must be same length."
            )
        if numeric_feats is not None and len(numeric_feats) != len(texts):
            raise ValueError(
                f"texts ({len(texts)}) and numeric_feats ({len(numeric_feats)}) "
                "must be same length."
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
            # RoBERTa: do NOT request token_type_ids — not needed
            # (AutoTokenizer will omit them automatically for roberta-*)
        )

        # squeeze the batch dimension added by return_tensors="pt"
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            # NOTE: no token_type_ids — RoBERTa docs confirm they are unused
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.numeric_feats is not None:
            item["numeric"] = torch.tensor(
                self.numeric_feats[idx], dtype=torch.float
            )

        return item

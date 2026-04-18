"""
model.py — Hybrid DistilBERT + numeric feature fusion classifier (v12).

Architecture
────────────
  DistilBertModel  →  last_hidden_state[:, 0, :]   (B, 768)   ← [CLS] token
  numeric features →  nn.Linear(2, 64) + GELU + LN  (B, 64)
  concat           →  (B, 832)
  classifier head  →  Dropout → Linear(832, 416) → GELU
                    → Dropout → Linear(416, 2)

Why last_hidden_state[:, 0, :] instead of pooler_output
────────────────────────────────────────────────────────
  DistilBertModel does NOT expose a pooler_output field.
  The HF DistilBERT docs confirm its forward() returns only
  last_hidden_state (and optionally hidden_states / attentions).
  The standard practice is to take the first token position (the
  [CLS] token) from last_hidden_state as the sequence representation.

  Compare to:
    BERT / RoBERTa  → pooler_output  (linear+tanh on [CLS])   ← v11
    DeBERTa-v3      → pooler_output  (linear+tanh on [CLS])   ← v8
    DistilBERT      → last_hidden_state[:, 0, :]  (raw [CLS]) ← v12

  We add a single LayerNorm after extracting [CLS] to compensate for
  the missing tanh activation that BERT's pooler provides.

Supported backbones
────────────────────
  distilbert-base-uncased   (default)
  distilbert-base-cased
  distilroberta-base        (768-dim, RoBERTa-style tokens)
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, AutoModel, AutoConfig


class DistilBertMatchupClassifier(nn.Module):
    """
    DistilBERT encoder + numeric projection → binary win/loss classifier.

    Parameters
    ----------
    model_name    : HF model id
    num_numeric   : numeric feature vector size (default 2)
    numeric_proj  : hidden dim for numeric projection (default 64)
    dropout       : dropout for classifier head (default 0.1)
    num_labels    : 2 for binary classification
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_numeric: int = 2,
        numeric_proj: int = 64,
        dropout: float = 0.1,
        num_labels: int = 2,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── DistilBERT backbone ──────────────────────────────────────────────
        # Use AutoModel so this also works for distilroberta-base
        self.distilbert = AutoModel.from_pretrained(model_name)
        hidden_size = self.distilbert.config.hidden_size  # 768 for all distil-base variants

        # ── [CLS] normalisation ──────────────────────────────────────────────
        # DistilBERT has no pooler (linear+tanh) so we add a LayerNorm to
        # stabilise the raw hidden state before fusion.
        self.cls_norm = nn.LayerNorm(hidden_size)

        # ── numeric feature projection ───────────────────────────────────────
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric, numeric_proj),
            nn.GELU(),
            nn.LayerNorm(numeric_proj),
        )

        # ── classifier head ──────────────────────────────────────────────────
        fused_size = hidden_size + numeric_proj  # 768 + 64 = 832
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_size, fused_size // 2),   # 832 → 416
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_size // 2, num_labels),   # 416 → 2
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
        numeric: torch.Tensor,         # (B, num_numeric)
        labels: torch.Tensor = None,   # (B,)  optional
        # NO token_type_ids — DistilBERT has no segment embeddings
    ):
        """
        Parameters
        ----------
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)
        numeric        : (B, num_numeric)
        labels         : (B,) — if provided, loss is computed and returned

        Returns
        -------
        dict with keys 'logits' and optionally 'loss'
        """
        # ── encode ───────────────────────────────────────────────────────────
        # DistilBertModel forward: input_ids + attention_mask only
        # Returns BaseModelOutput with .last_hidden_state (B, seq_len, H)
        encoder_out = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids: intentionally omitted — not supported
        )

        # Extract [CLS] token from position 0
        # Shape: (B, hidden_size)
        cls_hidden = encoder_out.last_hidden_state[:, 0, :]
        cls_hidden = self.cls_norm(cls_hidden)   # stabilise

        # ── numeric projection ────────────────────────────────────────────────
        numeric_out = self.numeric_proj(numeric)  # (B, 64)

        # ── fusion + classify ─────────────────────────────────────────────────
        fused = torch.cat([cls_hidden, numeric_out], dim=-1)   # (B, 832)
        logits = self.classifier(fused)                         # (B, 2)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


def build_model(model_name: str, num_numeric: int = 2) -> DistilBertMatchupClassifier:
    """Convenience factory used by trainer.py."""
    return DistilBertMatchupClassifier(
        model_name=model_name,
        num_numeric=num_numeric,
    )

"""
model.py — Hybrid RoBERTa + numeric feature fusion model for v11.

Architecture
------------
  RobertaForSequenceClassification is NOT used directly here because we
  want to fuse numeric features (seed gap, Massey gap) with the pooled
  [CLS] hidden state before the final classifier — mirroring v8's DeBERTa
  hybrid approach.

  Instead we use the bare RobertaModel and add our own classifier head:

    RobertaModel                  →  pooled_output  (B, 768)
    numeric features              →  nn.Linear       (B, 64)
    concat([pooled, numeric_proj]) → nn.Dropout
                                   → nn.Linear(768+64, 2)

  This lets the transformer learn from unstructured text while the
  numeric branch captures the hard seed/rating signal.

RoBERTa API notes from HF docs
--------------------------------
  • forward() accepts:  input_ids, attention_mask
  • Does NOT use token_type_ids (pass nothing / None)
  • pooler_output: the [CLS] hidden state after a linear+tanh layer
    — shape (batch_size, hidden_size)
  • hidden_size = 768 for roberta-base, 1024 for roberta-large
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, AutoConfig


class RobertaMatchupClassifier(nn.Module):
    """
    RoBERTa encoder + small numeric projection → binary classifier.

    Parameters
    ----------
    model_name    : HF model id  (e.g. 'FacebookAI/roberta-base')
    num_numeric   : size of the numeric feature vector (default 2)
    numeric_proj  : hidden size for the numeric projection layer
    dropout       : dropout probability for the classifier head
    num_labels    : 2 for binary win/loss
    """

    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-base",
        num_numeric: int = 2,
        numeric_proj: int = 64,
        dropout: float = 0.1,
        num_labels: int = 2,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── RoBERTa backbone ─────────────────────────────────────────────────
        # add_pooling_layer=True so we get pooler_output ([CLS] token rep)
        self.roberta = RobertaModel.from_pretrained(
            model_name, add_pooling_layer=True
        )
        hidden_size = self.roberta.config.hidden_size  # 768 or 1024

        # ── numeric feature projection ───────────────────────────────────────
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric, numeric_proj),
            nn.GELU(),
            nn.LayerNorm(numeric_proj),
        )

        # ── classifier head ──────────────────────────────────────────────────
        fused_size = hidden_size + numeric_proj
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_size, fused_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_size // 2, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric: torch.Tensor,
        labels: torch.Tensor = None,
        # token_type_ids intentionally omitted — RoBERTa ignores them
    ):
        """
        Parameters
        ----------
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)
        numeric        : (B, num_numeric)
        labels         : (B,) — optional, required for loss computation

        Returns
        -------
        dict with keys 'logits' and optionally 'loss'
        """
        # ── encode with RoBERTa ──────────────────────────────────────────────
        # RoBERTa forward: NO token_type_ids argument
        roberta_out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=None  ← do not pass; RoBERTa docs confirm unused
        )
        # pooler_output is the [<s>] token representation after linear+tanh
        # shape: (B, hidden_size)
        pooled = roberta_out.pooler_output

        # ── numeric projection ───────────────────────────────────────────────
        numeric_out = self.numeric_proj(numeric)        # (B, numeric_proj)

        # ── fusion + classify ────────────────────────────────────────────────
        fused = torch.cat([pooled, numeric_out], dim=-1)   # (B, hidden+proj)
        logits = self.classifier(fused)                     # (B, num_labels)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


def build_model(model_name: str, num_numeric: int = 2) -> RobertaMatchupClassifier:
    """Convenience factory used by trainer.py."""
    return RobertaMatchupClassifier(
        model_name=model_name,
        num_numeric=num_numeric,
    )

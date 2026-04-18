# v12 — Paris Text + DistilBERT

Port of the Paris-text pipeline to **DistilBERT** (`distilbert-base-uncased`).
DistilBERT is a distilled version of BERT — ~40% fewer parameters, ~60%
faster, retaining ~97% of BERT's performance on GLUE benchmarks.

## Model comparison across iterations

| | v8 DeBERTa | v11 RoBERTa | v12 DistilBERT |
|---|---|---|---|
| Backbone | deberta-v3-small | roberta-base | distilbert-base-uncased |
| Layers | 6 | 12 | **6** |
| Hidden size | 768 | 768 | 768 |
| Params | ~44M | ~125M | **~67M** |
| token_type_ids | optional | ✗ | ✗ |
| pooler_output | ✓ | ✓ | **✗** |
| CLS extraction | pooler_output | pooler_output | **last_hidden_state[:,0,:]** |
| Relative speed | 1× | 0.8× | **~2×** |
| GPU required | yes | yes | CPU feasible |

## Key DistilBERT API differences

### 1. No `token_type_ids`
DistilBERT removes the segment embedding layer entirely.
`AutoTokenizer` will not return `token_type_ids` for distilbert-* models.
They are never requested or forwarded in this codebase.

### 2. No `pooler_output`
`DistilBertModel.forward()` returns a `BaseModelOutput` with only
`last_hidden_state` — there is no `.pooler_output` attribute.
We extract the `[CLS]` token manually:

```python
cls_hidden = encoder_out.last_hidden_state[:, 0, :]   # (B, 768)
cls_hidden = self.cls_norm(cls_hidden)                 # stabilise
```

The `LayerNorm` after extraction compensates for the missing
`linear+tanh` pooler that BERT/RoBERTa apply.

## Architecture

```
text string
    │
    ▼
DistilBertTokenizerFast  →  input_ids + attention_mask only
    │
    ▼
DistilBertModel (6 layers)  →  last_hidden_state  (B, seq_len, 768)
                                      │
                               [:, 0, :]  ← [CLS] token
                                      │
                               LayerNorm  →  (B, 768)
    │
numeric [seed_gap, massey_gap]
    │
    ▼
Linear(2→64) + GELU + LayerNorm  →  (B, 64)
    │
concat  →  (B, 832)
    │
    ▼
Dropout → Linear(832→416) → GELU → Dropout → Linear(416→2)
```

## Run

```bash
# from repo root
python iterations/v12/src/main.py --data-dir data

# CPU is feasible for DistilBERT (slower but works)
python iterations/v12/src/main.py --data-dir data --cpu

# cased variant
python iterations/v12/src/main.py --data-dir data --model distilbert-base-cased

# RoBERTa-based distil (recommended if you want RoBERTa tokenisation)
python iterations/v12/src/main.py --data-dir data --model distilroberta-base

# train only, skip submission
python iterations/v12/src/main.py --data-dir data --no-submission
```

Outputs `submission_v12.csv` in repo root (or `--output-dir`).

## Hyperparameters (defaults)

| Param | Value | Note |
|---|---|---|
| model | `distilbert-base-uncased` | |
| max_len | 128 | matchup texts are short |
| epochs | 4 | |
| batch_size | 32 | larger than v11 — DistilBERT is lighter |
| lr | 3e-5 | slightly higher than BERT recommendation |
| warmup_ratio | 0.1 | |
| weight_decay | 0.01 | |
| grad_clip | 1.0 | |
| val_split | 10% | |

## Ensembling with v8 / v11

DistilBERT's speed makes it ideal for generating multiple seeds cheaply.
Consider averaging predictions:

```python
import pandas as pd
v8  = pd.read_csv("submission.csv").set_index("ID")        # DeBERTa
v11 = pd.read_csv("submission_v11.csv").set_index("ID")    # RoBERTa
v12 = pd.read_csv("submission_v12.csv").set_index("ID")    # DistilBERT

ensemble = (v8["Pred"] + v11["Pred"] + v12["Pred"]) / 3
ensemble.reset_index().to_csv("submission_ensemble.csv", index=False)
```

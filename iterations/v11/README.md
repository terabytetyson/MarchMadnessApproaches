# v11 — Paris Text + RoBERTa-base

Direct port of **v8** (Paris text + DeBERTa-v3-small) to **RoBERTa-base**
(`FacebookAI/roberta-base`).

## What changed from v8

| Aspect | v8 (DeBERTa-v3-small) | v11 (RoBERTa-base) |
|---|---|---|
| Backbone | `microsoft/deberta-v3-small` | `FacebookAI/roberta-base` |
| Tokenizer | SentencePiece (DebertaV2Tokenizer) | Byte-level BPE (RobertaTokenizerFast) |
| `token_type_ids` | used | **not used** — omitted entirely |
| Hidden size | 768 | 768 |
| Special tokens | `[CLS]` / `[SEP]` | `<s>` / `</s>` |
| GPU required | yes (reasonable speed) | yes (same) |

The key API difference: **RoBERTa has no `token_type_ids`**.  
The HF docs state: *"RoBERTa doesn't have `token_type_ids` so you don't
need to indicate which token belongs to which segment."*  
Passing them is silently ignored by the model, but they are not requested
from the tokenizer and not forwarded in any call in this codebase.

## Architecture

```
text string
    │
    ▼
RobertaTokenizerFast  →  input_ids + attention_mask (no token_type_ids)
    │
    ▼
RobertaModel (roberta-base)  →  pooler_output  (B, 768)
    │
numeric features [seed_gap, massey_gap]
    │
    ▼
nn.Linear(2→64) + GELU + LayerNorm  →  (B, 64)
    │
concat  →  (B, 832)
    │
    ▼
Dropout → Linear(832→416) → GELU → Dropout → Linear(416→2)
    │
    ▼
CrossEntropyLoss / Softmax
```

## Run

```bash
# from repo root
python iterations/v11/src/main.py --data-dir data

# CPU-only
python iterations/v11/src/main.py --data-dir data --cpu

# roberta-large (needs more VRAM, ~2× slower)
python iterations/v11/src/main.py --data-dir data --model FacebookAI/roberta-large

# train only, no submission
python iterations/v11/src/main.py --data-dir data --no-submission
```

Outputs `submission_v11.csv` in the repo root (or `--output-dir`).

## Hyperparameters (defaults)

| Param | Value |
|---|---|
| model | `FacebookAI/roberta-base` |
| max_len | 128 |
| epochs | 4 |
| batch_size | 16 |
| lr | 2e-5 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| grad_clip | 1.0 |
| val_split | 10% |

## Compared to DeBERTa (v8)

RoBERTa-base trains faster per step (simpler attention mechanism,
no disentangled attention). DeBERTa-v3-small often edges it slightly on
NLU benchmarks, but RoBERTa's larger pretraining corpus can give an edge
on short domain-specific texts like these matchup descriptions.
Recommend running both and ensembling.

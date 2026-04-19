# March Madness Approaches

Kaggle data in `data/`, plus five transformer and ML pipelines for predicting NCAA Tournament outcomes.

## Iterations

| Directory | Model | Backbone | Notes |
| --- | --- | --- | --- |
| `iterations/v8` | Paris text + **DeBERTa-v3-small** | `microsoft/deberta-v3-small` | Men's only; needs GPU |
| `iterations/v9` | Paris features + **ExtraTrees** | — | Men's + women's; no GPU required |
| `iterations/v10` | **Kalshi** progression + simulation | — | Men's; uses `goto-conversion` |
| `iterations/v11` | Paris text + **RoBERTa-base** | `FacebookAI/roberta-base` | Men's only; needs GPU |
| `iterations/v12` | Paris text + **DistilBERT** | `distilbert-base-uncased` | Men's only; CPU feasible |

### Transformer model comparison (v8 / v11 / v12)

| | v8 DeBERTa | v11 RoBERTa | v12 DistilBERT |
| --- | --- | --- | --- |
| Layers | 6 | 12 | 6 |
| Hidden size | 768 | 768 | 768 |
| `token_type_ids` | optional | ✗ | ✗ |
| `pooler_output` | ✓ | ✓ | ✗ — uses `last_hidden_state[:,0,:]` |
| Relative speed | 1× | ~0.8× | ~2× |
| GPU required | yes | yes | no (slower on CPU) |

All three transformer iterations use the same Paris-style matchup text format and a hybrid numeric fusion head (seed gap + Massey rating gap).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For v8/v9/v11/v12, optionally add `data/cbb_round1_odds_all_champions_2002_2025.json` or set `ODDS_PATH` (see `data/OPTIONAL_ODDS_FOR_TRAINING.txt`).

`MMasseyOrdinals` is split into `data/MMasseyOrdinals_part1.csv` (seasons ≤ 2016) and `data/MMasseyOrdinals_part2.csv` (≥ 2017). A single `MMasseyOrdinals.csv` also works if you have it locally.

---

## Run

All commands are run from the repo root with `DATA_DIR` pointing at `data/` (the default).

**ExtraTrees — v9** *(no GPU needed)*
```bash
python iterations/v9/src/main.py --data-dir data
```

**DeBERTa — v8**
```bash
python iterations/v8/src/main.py --data-dir data
python iterations/v8/src/main.py --data-dir data --cpu   # add --cpu if no GPU
```

**Kalshi — v10**
```bash
python iterations/v10/main.py --data-dir data --n-sims 50000
```

**RoBERTa — v11**
```bash
python iterations/v11/src/main.py --data-dir data
python iterations/v11/src/main.py --data-dir data --cpu
python iterations/v11/src/main.py --data-dir data --model FacebookAI/roberta-large
```

**DistilBERT — v12** *(lightest transformer; CPU is feasible)*
```bash
python iterations/v12/src/main.py --data-dir data
python iterations/v12/src/main.py --data-dir data --cpu
python iterations/v12/src/main.py --data-dir data --model distilroberta-base
```

---

## Outputs

| Iteration | Output file |
| --- | --- |
| v8 | `submission.csv` |
| v9 | `submission.csv` |
| v10 | `output_kalshi_mens.csv` |
| v11 | `submission_v11.csv` |
| v12 | `submission_v12.csv` |

All output files are written to the repo root by default. Override with `--output-dir`.

---

## Ensembling

The three transformer iterations (v8, v11, v12) can be averaged for a simple ensemble:

```python
import pandas as pd

v8  = pd.read_csv("submission.csv").set_index("ID")
v11 = pd.read_csv("submission_v11.csv").set_index("ID")
v12 = pd.read_csv("submission_v12.csv").set_index("ID")

ensemble = (v8["Pred"] + v11["Pred"] + v12["Pred"]) / 3
ensemble.reset_index().to_csv("submission_ensemble.csv", index=False)
```

---

## Environment overrides

| Variable | Used by | Default |
| --- | --- | --- |
| `DATA_DIR` | all | `data` |
| `ODDS_PATH` | v8, v9, v11, v12 | — |
| `KALSHI_PATH` | v10 | — |
| `SEASON` | all | `2025` |
| `N_SIMS` | v10 | `50000` |

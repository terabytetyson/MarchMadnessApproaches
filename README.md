# March Madness approaches (Paris, DeBERTa, Kalshi)

Clean extract: **Kaggle data** in `data/`, plus three pipelines:

| Directory | Model | Notes |
|-----------|--------|--------|
| `iterations/v9` | Paris features + **ExtraTrees** | Men's + women's; no GPU required |
| `iterations/v8` | Paris text + **DeBERTa-v3-small** | Men's only; needs GPU for reasonable train time |
| `iterations/v10` | **Kalshi** progression + simulation | Men's; uses `goto-conversion` |

## Setup

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For v8/v9, add `data/cbb_round1_odds_all_champions_2002_2025.json` or set `ODDS_PATH` (see `data/OPTIONAL_ODDS_FOR_TRAINING.txt`).

## Run

From repo root, with `DATA_DIR` pointing at `data/` (default if you run from this tree):

**Paris (v9)**

```bash
python iterations/v9/src/main.py --data-dir data
```

**DeBERTa (v8)** — v8 is the DeBERTa-v3-small Paris-text model used in the course work.

```bash
python iterations/v8/src/main.py --data-dir data --cpu   # add --cpu if no GPU
```

**Kalshi (v10)**

```bash
python iterations/v10/main.py --data-dir data --n-sims 50000
```

Outputs go to the repo root by default (`output_kalshi_mens.csv` for v10; v8/v9 write `submission.csv` when generating).

Environment overrides: `DATA_DIR`, `ODDS_PATH`, `KALSHI_PATH`, `SEASON`, `N_SIMS`.

"""
v12 — Paris text + DistilBERT sequence classifier
Mirrors v11 (RoBERTa) but swaps in DistilBERT.

Key DistilBERT differences vs RoBERTa / DeBERTa:
  • No token_type_ids  (same as RoBERTa)
  • No pooler_output   — use last_hidden_state[:, 0, :] ([CLS] vector)
  • 6 transformer layers instead of 12 → ~2× faster, ~40% fewer params
  • Hidden size 768 (same as roberta-base / deberta-v3-small)
  • Model class: DistilBertModel (not ForSequenceClassification — we
    add our own fusion head, same pattern as v11)

Usage (from repo root):
    python iterations/v12/src/main.py --data-dir data
    python iterations/v12/src/main.py --data-dir data --cpu
    python iterations/v12/src/main.py --data-dir data --model distilbert-base-uncased
    python iterations/v12/src/main.py --data-dir data --model distilroberta-base
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from features import build_matchup_texts, load_and_prepare_data
from trainer import run_training
from predictor import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="v12: Paris text + DistilBERT")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--model", default="distilbert-base-uncased",
                   help="HF model id. Also works with 'distilroberta-base' or "
                        "'distilbert-base-cased'")
    p.add_argument("--max-len", type=int, default=128,
                   help="Max token sequence length (DistilBERT cap is 512)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32,
                   help="DistilBERT is lighter so larger batches fit easily")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Slightly higher LR than BERT-base is typical for DistilBERT")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "."))
    p.add_argument("--season", type=int,
                   default=int(os.environ.get("SEASON", 2025)))
    p.add_argument("--no-submission", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=== v12  Paris text + DistilBERT ===")
    log.info("model        : %s", args.model)
    log.info("data_dir     : %s", args.data_dir)
    log.info("season       : %s", args.season)
    log.info("max_len      : %s", args.max_len)
    log.info("epochs       : %s", args.epochs)
    log.info("batch_size   : %s", args.batch_size)
    log.info("lr           : %s", args.lr)

    # ── 1. load data & build Paris-style matchup texts ───────────────────────
    log.info("Loading and preparing data …")
    train_df, seed_df, teams_df, ordinals_df = load_and_prepare_data(args.data_dir)

    log.info("Building matchup text features …")
    train_texts, train_labels, train_numeric, _ = build_matchup_texts(
        train_df, seed_df, teams_df, ordinals_df, split="train"
    )
    log.info("  training pairs: %d", len(train_texts))

    # ── 2. fine-tune DistilBERT ──────────────────────────────────────────────
    model, tokenizer = run_training(
        texts=train_texts,
        labels=train_labels,
        numeric_features=train_numeric,
        model_name=args.model,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_cpu=args.cpu,
    )

    # ── 3. generate submission ───────────────────────────────────────────────
    if not args.no_submission:
        log.info("Generating submission for season %d …", args.season)
        test_texts, _, test_numeric, matchup_ids = build_matchup_texts(
            train_df, seed_df, teams_df, ordinals_df,
            split="test", season=args.season
        )
        generate_submission(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            numeric_features=test_numeric,
            matchup_ids=matchup_ids,
            max_len=args.max_len,
            output_dir=args.output_dir,
            use_cpu=args.cpu,
        )
    else:
        log.info("--no-submission set; skipping prediction step.")

    log.info("Done.")


if __name__ == "__main__":
    main()

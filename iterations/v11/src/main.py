"""
v11 — Paris text + RoBERTa-base sequence classifier
Mirrors v8 (DeBERTa-v3-small) but swaps in RoBERTa-base.

Key RoBERTa differences vs DeBERTa-v3-small:
  • No token_type_ids — RoBERTa ignores segment ids entirely
  • Byte-level BPE tokenizer (FacebookAI/roberta-base)
  • pad_token_id=1, bos=0, eos=2  (<pad>, <s>, </s>)
  • Hidden size 768 (roberta-base) — same as deberta-v3-small
  • RobertaForSequenceClassification is the classification head

Usage (from repo root):
    python iterations/v11/src/main.py --data-dir data
    python iterations/v11/src/main.py --data-dir data --cpu
    python iterations/v11/src/main.py --data-dir data --model FacebookAI/roberta-large
"""

import argparse
import os
import sys
import logging

# make sibling modules importable when run directly
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
    p = argparse.ArgumentParser(description="v11: Paris text + RoBERTa")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"),
                   help="Path to Kaggle data directory")
    p.add_argument("--model", default="FacebookAI/roberta-base",
                   help="HF model id, e.g. FacebookAI/roberta-base or FacebookAI/roberta-large")
    p.add_argument("--max-len", type=int, default=128,
                   help="Max token sequence length (roberta-base cap is 512)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "."),
                   help="Where to write submission.csv")
    p.add_argument("--season", type=int,
                   default=int(os.environ.get("SEASON", 2025)),
                   help="Tournament season to generate predictions for")
    p.add_argument("--no-submission", action="store_true",
                   help="Skip submission generation (train only)")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=== v11  Paris text + RoBERTa ===")
    log.info("model        : %s", args.model)
    log.info("data_dir     : %s", args.data_dir)
    log.info("season       : %s", args.season)
    log.info("max_len      : %s", args.max_len)
    log.info("epochs       : %s", args.epochs)
    log.info("batch_size   : %s", args.batch_size)
    log.info("lr           : %s", args.lr)

    # ── 1. load data & build Paris-style matchup texts ──────────────────────
    log.info("Loading and preparing data …")
    train_df, seed_df, teams_df, ordinals_df = load_and_prepare_data(args.data_dir)

    log.info("Building matchup text features …")
    train_texts, train_labels, train_numeric, _ = build_matchup_texts(
        train_df, seed_df, teams_df, ordinals_df, split="train"
    )
    log.info("  training pairs: %d", len(train_texts))

    # ── 2. fine-tune RoBERTa ─────────────────────────────────────────────────
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

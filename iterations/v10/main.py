"""V10: Men's Kalshi (simulation + goto_conversion fallbacks)."""

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))

from src import config
from src.kalshi_pipeline import generate_kalshi_submission


def main():
    parser = argparse.ArgumentParser(description="Kalshi men's pipeline (Monte Carlo + progression)")
    parser.add_argument("--kalshi", default=config.KALSHI_PATH, help="Kalshi progression JSON")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Kaggle data directory")
    parser.add_argument("--season", type=int, default=config.SEASON, help="Tournament season year")
    parser.add_argument("--n-sims", type=int, default=config.N_SIMS, help="Bracket simulations")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--stage", default="Stage1", choices=["Stage1", "Stage2"])
    args = parser.parse_args()

    out = args.output or os.path.join(config.OUTPUT_DIR, "output_kalshi_mens.csv")
    generate_kalshi_submission(
        kalshi_path=args.kalshi,
        data_dir=args.data_dir,
        season=args.season,
        n_sims=args.n_sims,
        output_path=out,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()

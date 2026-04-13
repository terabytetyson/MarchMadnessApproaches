"""V9: Classic Paris (box + style) + market odds. ExtraTrees. No BERT. Men's + women's."""

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
import deps  # noqa: F401
from src import config
from src.build_training_data import build_training_data, print_samples
from src.train_model import train
from src.generate_submission import generate_submission, evaluate


def main():
    parser = argparse.ArgumentParser(description="V9 Paris: classic raddar approach + market odds")
    parser.add_argument("--data-dir", default=None, help=f"Data path (default: {config.DATA_DIR})")
    parser.add_argument("--odds-path", default=None, help=f"Odds JSON (default: {config.ODDS_PATH})")
    parser.add_argument("--model-path", default=None, help=f"Model path (default: {config.MODEL_DIR}/paris-v9)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--stage", default="Stage1", choices=["Stage1", "Stage2"])
    parser.add_argument("--eval", action="store_true", help="Eval mode")
    parser.add_argument("--test-years", default=None, help="Comma-separated test years")
    parser.add_argument("--clip-extreme", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or config.DATA_DIR
    odds_path = args.odds_path or config.ODDS_PATH
    model_path = args.model_path or os.path.join(config.MODEL_DIR, "paris-v9")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    if not os.path.exists(odds_path):
        print(f"Error: Odds file not found: {odds_path}")
        sys.exit(1)

    model_exists = os.path.exists(os.path.join(model_path, "paris_model.pkl"))

    if args.eval:
        test_years = [int(y) for y in args.test_years.split(",")] if args.test_years else config.TEST_YEARS
        print(f"Eval mode: test years = {test_years}")
        if not args.skip_train:
            train(
                data_dir=data_dir,
                odds_path=odds_path,
                output_dir=model_path,
                exclude_seasons=test_years,
                val_seasons=config.VAL_YEARS if config.VAL_YEARS and all(v in test_years for v in config.VAL_YEARS) else None,
            )
        elif not model_exists:
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        evaluate(
            data_dir=data_dir,
            odds_path=odds_path,
            model_path=model_path,
            test_years=test_years,
            clip_extreme=args.clip_extreme,
        )
    else:
        if not args.skip_train:
            train(
                data_dir=data_dir,
                odds_path=odds_path,
                output_dir=model_path,
            )
        elif not model_exists:
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        print("Generating submission...")
        generate_submission(
            data_dir=data_dir,
            odds_path=odds_path,
            model_path=model_path,
            output_path=os.path.join(config.OUTPUT_DIR, "submission.csv"),
            stage=args.stage,
        )
    print("Done.")


if __name__ == "__main__":
    main()

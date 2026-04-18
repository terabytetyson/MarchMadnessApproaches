"""Train all 3 BERT variants: base, aggressive, conservative."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iterations.v11_bert_base import config as config_base
from iterations.v11_bert_aggressive import config as config_agg
from iterations.v11_bert_conservative import config as config_cons
from bert_trainer import train_bert


def main():
    parser = argparse.ArgumentParser(description="Train all 3 BERT variants")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--odds-path", default=None, help="Odds JSON path")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    parser.add_argument("--skip-base", action="store_true", help="Skip base variant")
    parser.add_argument("--skip-agg", action="store_true", help="Skip aggressive variant")
    parser.add_argument("--skip-cons", action="store_true", help="Skip conservative variant")
    args = parser.parse_args()

    configs = [
        (config_base, "base", not args.skip_base),
        (config_agg, "aggressive", not args.skip_agg),
        (config_cons, "conservative", not args.skip_cons),
    ]

    for cfg, name, should_run in configs:
        if should_run:
            try:
                train_bert(
                    config=cfg,
                    data_dir=args.data_dir,
                    odds_path=args.odds_path,
                    use_cpu=args.cpu,
                )
            except Exception as e:
                print(f"Error training {name} variant: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Skipping {name} variant")

    print("\nAll BERT variants trained successfully!")


if __name__ == "__main__":
    main()

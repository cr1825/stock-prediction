"""Command-line entry point.

Examples
--------
    python main.py --ticker AAPL
    python main.py --ticker TSLA --start 2015-01-01 --epochs 80 --lookback 90
    python main.py --synthetic --epochs 3      # offline smoke test (fake data!)
"""

import argparse
import logging

from src.config import Config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description="Train an LSTM stock predictor on yfinance data.")
    p.add_argument("--ticker", default=d.ticker, help="Yahoo Finance symbol, e.g. AAPL, TSLA, ^GSPC")
    p.add_argument("--start", default=d.start, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=d.end, help="End date YYYY-MM-DD (default: latest)")
    p.add_argument("--lookback", type=int, default=d.lookback, help="Days of history per sample")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--units", type=int, nargs=2, default=list(d.lstm_units),
                   metavar=("L1", "L2"), help="Units in the two LSTM layers")
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--synthetic", action="store_true",
                   help="Use generated random-walk data (offline smoke test only)")
    return p.parse_args()


def print_summary(result: dict) -> None:
    m, f = result["metrics"], result["forecast"]
    model, naive, bt = m["model"], m["naive_persistence_baseline"], m["backtest"]
    print("\n" + "=" * 64)
    print(f" {result['ticker']} - test-set results ({m['n_test_samples']} days, "
          f"{m['split_dates']['test_start']} -> {m['split_dates']['test_end']})")
    print("=" * 64)
    print(f" {'Metric':<26}{'LSTM':>14}{'Naive (prev close)':>22}")
    print(f" {'RMSE':<26}{model['rmse']:>14.4f}{naive['rmse']:>22.4f}")
    print(f" {'MAE':<26}{model['mae']:>14.4f}{naive['mae']:>22.4f}")
    print(f" {'MAPE (%)':<26}{model['mape']:>14.3f}{naive['mape']:>22.3f}")
    print(f" {'Directional accuracy (%)':<26}{model['directional_accuracy']:>14.2f}"
          f"{m['always_up_directional_accuracy']:>22.2f}  <- 'always up'")
    print(f"\n RMSE vs naive baseline: {m['rmse_improvement_vs_naive_pct']:+.2f}%")
    print(f" Long/flat backtest return: {bt['strategy_total_return_pct']:+.1f}%  "
          f"(buy & hold: {bt['buy_and_hold_total_return_pct']:+.1f}%, no costs)")
    print(f"\n Next-day forecast for {f['forecast_date']}: {f['direction']} "
          f"({f['predicted_return_pct']:+.2f}%) -> ${f['predicted_close']:.2f} "
          f"(last close ${f['last_close']:.2f} on {f['as_of_date']})")
    print(" Educational project - NOT financial advice.")
    print("=" * 64)
    print("\nSaved files:")
    for k, v in result["paths"].items():
        print(f"  {k:<18} {v}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end, lookback=args.lookback,
        epochs=args.epochs, batch_size=args.batch_size, lstm_units=tuple(args.units),
        dropout=args.dropout, learning_rate=args.lr, seed=args.seed,
        synthetic=args.synthetic,
    )
    print_summary(run_pipeline(cfg))


if __name__ == "__main__":
    main()

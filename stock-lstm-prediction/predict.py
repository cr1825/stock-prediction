"""Forecast the next trading day using a previously trained model.

    python predict.py --ticker AAPL
"""

import argparse
import logging

from src.pipeline import forecast_with_saved_model


def main() -> None:
    p = argparse.ArgumentParser(description="Next-day forecast from a saved LSTM model.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--model-dir", default="models")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    f = forecast_with_saved_model(args.ticker, args.model_dir)
    print(f"\n{args.ticker.upper()}: last close ${f['last_close']:.2f} on {f['as_of_date']}")
    print(f"Forecast for {f['forecast_date']}: {f['direction']} "
          f"({f['predicted_return_pct']:+.2f}%) -> ${f['predicted_close']:.2f}")
    print("Educational project - NOT financial advice.\n")


if __name__ == "__main__":
    main()

"""Evaluation: error metrics, directional accuracy, naive baseline, backtest.

Avoid claims like "95% accurate". Stock prices are highly autocorrelated, so
even "tomorrow = today" gets a tiny price error. A model is only interesting if
it beats that naive baseline, so we always report both.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err / actual)) * 100),  # percent
    }


def directional_accuracy(actual_return, predicted_return) -> float:
    """Share (%) of days where the predicted direction (up/down) was correct."""
    actual_up = np.asarray(actual_return) > 0
    pred_up = np.asarray(predicted_return) > 0
    return float(np.mean(actual_up == pred_up) * 100)


def evaluate_predictions(pred_df: pd.DataFrame) -> Dict:
    """``pred_df`` needs: prev_close, actual_close, actual_return,
    predicted_return, predicted_close."""
    model = regression_metrics(pred_df["actual_close"], pred_df["predicted_close"])
    model["directional_accuracy"] = directional_accuracy(
        pred_df["actual_return"], pred_df["predicted_return"]
    )

    # Baseline 1: persistence ("tomorrow's price = today's price")
    naive = regression_metrics(pred_df["actual_close"], pred_df["prev_close"])
    # Baseline 2: always predict "up"
    always_up = float(np.mean(pred_df["actual_return"] > 0) * 100)

    return {
        "n_test_samples": int(len(pred_df)),
        "model": model,
        "naive_persistence_baseline": naive,
        "always_up_directional_accuracy": always_up,
        "rmse_improvement_vs_naive_pct": float(
            (naive["rmse"] - model["rmse"]) / naive["rmse"] * 100
        ),
    }


def backtest_long_flat(pred_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Toy backtest: hold the stock when the model predicts 'up', else stay in cash.

    Ignores transaction costs and slippage - educational only.
    """
    position = (pred_df["predicted_return"] > 0).astype(float)
    strat_ret = position * pred_df["actual_return"]
    bh_ret = pred_df["actual_return"]

    curve = pd.DataFrame(
        {
            "strategy": np.exp(strat_ret.cumsum()),
            "buy_and_hold": np.exp(bh_ret.cumsum()),
        },
        index=pred_df.index,
    )

    def sharpe(r: pd.Series) -> float:
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    stats = {
        "strategy_total_return_pct": float((curve["strategy"].iloc[-1] - 1) * 100),
        "buy_and_hold_total_return_pct": float((curve["buy_and_hold"].iloc[-1] - 1) * 100),
        "strategy_sharpe": sharpe(strat_ret),
        "buy_and_hold_sharpe": sharpe(bh_ret),
        "days_in_market_pct": float(position.mean() * 100),
    }
    return stats, curve

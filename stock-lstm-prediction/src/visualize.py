"""Static (matplotlib) charts saved to the outputs folder."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .features import compute_macd, compute_rsi

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_price_overview(df: pd.DataFrame, ticker: str, path, last_n: int = 500) -> str:
    """Price + SMAs + Bollinger Bands, RSI and MACD panels."""
    d = df.copy()
    close = d["Close"]
    d["sma20"], d["sma50"] = close.rolling(20).mean(), close.rolling(50).mean()
    std = close.rolling(20).std()
    d["bb_up"], d["bb_lo"] = d["sma20"] + 2 * std, d["sma20"] - 2 * std
    d["rsi"] = compute_rsi(close)
    d["macd"], d["macd_sig"], d["macd_hist"] = compute_macd(close)
    d = d.iloc[-last_n:]

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1, 1.3]})
    a1.plot(d.index, d["Close"], label="Close", color="#1f77b4", lw=1.4)
    a1.plot(d.index, d["sma20"], label="SMA 20", color="#ff7f0e", lw=1)
    a1.plot(d.index, d["sma50"], label="SMA 50", color="#2ca02c", lw=1)
    a1.fill_between(d.index, d["bb_lo"], d["bb_up"], color="grey", alpha=0.15,
                    label="Bollinger (20, 2σ)")
    a1.set_title(f"{ticker} - price & technical indicators (last {len(d)} days)")
    a1.set_ylabel("Price"); a1.legend(loc="upper left", ncol=4)

    a2.plot(d.index, d["rsi"], color="#9467bd", lw=1)
    a2.axhline(70, color="red", ls="--", lw=0.8); a2.axhline(30, color="green", ls="--", lw=0.8)
    a2.set_ylabel("RSI (14)"); a2.set_ylim(0, 100)

    a3.plot(d.index, d["macd"], label="MACD", lw=1)
    a3.plot(d.index, d["macd_sig"], label="Signal", lw=1)
    a3.bar(d.index, d["macd_hist"], color="grey", alpha=0.5, label="Histogram")
    a3.set_ylabel("MACD"); a3.legend(loc="upper left", ncol=3)
    return _save(fig, path)


def plot_predictions(pred_df: pd.DataFrame, ticker: str, path, zoom_days: int = 60) -> str:
    """Actual vs predicted close on the test set (full range + zoomed view)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
    for ax, d, title in (
        (a1, pred_df, f"{ticker} - LSTM one-day-ahead predictions (test set)"),
        (a2, pred_df.iloc[-zoom_days:], f"Zoom: last {min(zoom_days, len(pred_df))} test days"),
    ):
        ax.plot(d.index, d["actual_close"], label="Actual close", color="#1f77b4", lw=1.6)
        ax.plot(d.index, d["predicted_close"], label="LSTM prediction",
                color="#d62728", lw=1.3, ls="--")
        ax.set_title(title); ax.set_ylabel("Price"); ax.legend(loc="upper left")
    return _save(fig, path)


def plot_training_history(history: dict, path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber loss (scaled returns)")
    ax.set_title("Training history"); ax.legend()
    return _save(fig, path)


def plot_return_scatter(pred_df: pd.DataFrame, directional_acc: float, path) -> str:
    """Predicted vs actual daily returns - shows how much signal the model has."""
    fig, ax = plt.subplots(figsize=(6, 6))
    a, p = pred_df["actual_return"] * 100, pred_df["predicted_return"] * 100
    ax.scatter(a, p, s=10, alpha=0.5)
    lim = max(a.abs().max(), p.abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls="--", label="Perfect prediction")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Actual next-day return (%)"); ax.set_ylabel("Predicted next-day return (%)")
    ax.set_title(f"Predicted vs actual returns\nDirectional accuracy: {directional_acc:.1f}%")
    ax.legend()
    return _save(fig, path)


def plot_backtest(curve: pd.DataFrame, ticker: str, path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(curve.index, curve["buy_and_hold"], label="Buy & hold", color="grey")
    ax.plot(curve.index, curve["strategy"], label="LSTM long/flat strategy", color="#d62728")
    ax.set_title(f"{ticker} - growth of $1 on the test period (no costs)")
    ax.set_ylabel("Portfolio value"); ax.legend()
    return _save(fig, path)

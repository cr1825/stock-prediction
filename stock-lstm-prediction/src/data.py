"""Data collection: download OHLCV data from Yahoo Finance (yfinance)."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep OHLCV only, strip timezone, sort, de-duplicate and drop bad rows."""
    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df["Volume"] = df["Volume"].fillna(0)
    return df


def fetch_stock_data(
    ticker: str,
    start: str = "2013-01-01",
    end: Optional[str] = None,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download daily split/dividend-adjusted OHLCV data with yfinance.

    The downloaded data is cached to ``<cache_dir>/<TICKER>.csv``. If the
    download fails (e.g. no internet) the cached copy is used instead.
    """
    import yfinance as yf

    ticker = ticker.upper().strip()
    cache_path = Path(cache_dir) / f"{ticker}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame()
    try:
        raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    except Exception as exc:  # network errors, rate limits, ...
        logger.warning("yfinance download failed for %s: %s", ticker, exc)

    if raw is None or raw.empty:
        if cache_path.exists():
            logger.warning("Using cached data for %s from %s", ticker, cache_path)
            return _clean(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol, the date "
            "range and your internet connection."
        )

    df = _clean(raw)
    df.to_csv(cache_path)
    logger.info("Downloaded %d rows for %s (%s -> %s)", len(df), ticker,
                df.index[0].date(), df.index[-1].date())
    return df


def generate_synthetic_data(n_days: int = 1500, seed: int = 0,
                            start_price: float = 100.0) -> pd.DataFrame:
    """Generate random-walk OHLCV data.

    ONLY for unit tests / offline smoke tests. Never report results from it.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.015, n_days)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df

"""Feature engineering: technical indicators computed from OHLCV data.

Every feature at row ``t`` only uses data up to and including day ``t``
(no look-ahead). Features are also expressed as ratios / returns so they are
roughly stationary, which is far friendlier for neural networks than raw prices.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "hl_range",
    "oc_change",
    "volume_change",
    "volume_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "close_to_sma_10",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_ema_12",
    "close_to_ema_26",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "bb_width",
    "bb_position",
    "momentum_10",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing), range 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    flat = (avg_gain == 0) & (avg_loss == 0)
    return rsi.where(~flat, 50.0)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd, macd_signal, macd - macd_signal


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with all columns in ``FEATURE_COLUMNS`` appended."""
    out = df.copy()
    close, high, low, open_, volume = (
        out["Close"], out["High"], out["Low"], out["Open"], out["Volume"]
    )

    # Price action
    out["log_return"] = np.log(close / close.shift(1))
    out["hl_range"] = (high - low) / close
    out["oc_change"] = (close - open_) / open_
    out["momentum_10"] = close / close.shift(10) - 1

    # Volume (indices such as ^GSPC may have no volume -> features become 0)
    if (volume > 0).any():
        vol = volume.replace(0, np.nan)
        out["volume_change"] = np.log(vol / vol.shift(1))
        out["volume_ratio"] = vol / vol.rolling(20).mean()
    else:
        out["volume_change"] = 0.0
        out["volume_ratio"] = 1.0

    # RSI (scaled to 0-1)
    out["rsi_14"] = compute_rsi(close, 14) / 100.0

    # MACD (normalised by price so it is comparable across time)
    macd, macd_signal, macd_hist = compute_macd(close)
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close
    out["macd_hist"] = macd_hist / close

    # Trend: distance of price from moving averages
    for w in (10, 20, 50):
        out[f"close_to_sma_{w}"] = close / close.rolling(w).mean() - 1
    for span in (12, 26):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"close_to_ema_{span}"] = close / ema - 1

    # Rolling standard deviation of returns (volatility)
    for w in (5, 10, 20):
        out[f"volatility_{w}"] = out["log_return"].rolling(w).std()

    # Bollinger Bands (20d, 2 std)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / sma20
    out["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)

"""Preprocessing: target creation, chronological split, scaling, sequences.

Data-integrity rules enforced here:
  * The split is strictly chronological (train < validation < test).
  * Scalers are fitted on the TRAINING rows only (no leakage from the future).
  * The target is the NEXT day's log return; the model never sees it as input.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

CLIP_VALUE = 10.0  # clip standardised features to +/- 10 std (outlier safety)


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame  # index = predicted day; prev_close, actual_close, actual_return
    scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: List[str]
    latest_window: np.ndarray  # (1, lookback, n_features) -> next-day forecast
    last_close: float
    last_date: pd.Timestamp
    split_dates: dict


def create_sequences(features: np.ndarray, targets: np.ndarray,
                     lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a (n, n_features) array into overlapping (lookback, n_features) windows.

    Window ``i`` covers rows ``i-lookback+1 .. i`` and is paired with ``targets[i]``.
    """
    idx = np.arange(lookback - 1, len(features))
    X = np.stack([features[i - lookback + 1: i + 1] for i in idx])
    return X.astype(np.float32), targets[idx].astype(np.float32), idx


def scale_features(df: pd.DataFrame, scaler: StandardScaler,
                   feature_columns: List[str]) -> np.ndarray:
    scaled = scaler.transform(df[feature_columns].values)
    return np.clip(scaled, -CLIP_VALUE, CLIP_VALUE)


def build_latest_window(df_feat: pd.DataFrame, scaler: StandardScaler,
                        feature_columns: List[str], lookback: int) -> np.ndarray:
    """Most recent ``lookback`` rows, scaled -> input for the next-day forecast."""
    valid = df_feat.dropna(subset=feature_columns)
    if len(valid) < lookback:
        raise ValueError(f"Need at least {lookback} valid rows, got {len(valid)}.")
    window = scale_features(valid.iloc[-lookback:], scaler, feature_columns)
    return window[np.newaxis, ...].astype(np.float32)


def prepare_data(df_feat: pd.DataFrame, lookback: int = 60, train_frac: float = 0.70,
                 val_frac: float = 0.15,
                 feature_columns: List[str] = None) -> PreparedData:
    feature_columns = list(feature_columns or FEATURE_COLUMNS)

    df = df_feat.dropna(subset=feature_columns).copy()
    # Target: log return from today's close to tomorrow's close.
    df["target"] = df["log_return"].shift(-1)
    df["next_close"] = df["Close"].shift(-1)

    labeled = df.dropna(subset=["target"])
    n = len(labeled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train <= lookback or n - n_train - n_val <= 20:
        raise ValueError(
            f"Not enough data ({n} usable rows) for lookback={lookback}. "
            "Use an earlier start date or a smaller lookback."
        )

    # Fit scalers on training rows ONLY.
    scaler = StandardScaler().fit(labeled[feature_columns].iloc[:n_train].values)
    target_scaler = StandardScaler().fit(labeled[["target"]].iloc[:n_train].values)

    X_all = scale_features(labeled, scaler, feature_columns)
    y_all = target_scaler.transform(labeled[["target"]].values).ravel()

    X, y, idx = create_sequences(X_all, y_all, lookback)

    train_mask = idx < n_train
    val_mask = (idx >= n_train) & (idx < n_train + n_val)
    test_mask = idx >= n_train + n_val

    test_rows = labeled.iloc[idx[test_mask]]
    # Date of the day being predicted = the date of the next row in the full frame.
    all_dates = df.index
    pos = all_dates.get_indexer(test_rows.index)
    predicted_dates = all_dates[pos + 1]

    test_meta = pd.DataFrame(
        {
            "prev_close": test_rows["Close"].values,
            "actual_close": test_rows["next_close"].values,
            "actual_return": test_rows["target"].values,
        },
        index=predicted_dates,
    )
    test_meta.index.name = "Date"

    split_dates = {
        "train_start": str(labeled.index[idx[train_mask][0]].date()),
        "train_end": str(labeled.index[idx[train_mask][-1]].date()),
        "val_start": str(labeled.index[idx[val_mask][0]].date()),
        "val_end": str(labeled.index[idx[val_mask][-1]].date()),
        "test_start": str(labeled.index[idx[test_mask][0]].date()),
        "test_end": str(labeled.index[idx[test_mask][-1]].date()),
    }

    return PreparedData(
        X_train=X[train_mask], y_train=y[train_mask],
        X_val=X[val_mask], y_val=y[val_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        test_meta=test_meta,
        scaler=scaler, target_scaler=target_scaler,
        feature_columns=feature_columns,
        latest_window=build_latest_window(df_feat, scaler, feature_columns, lookback),
        last_close=float(df["Close"].iloc[-1]),
        last_date=df.index[-1],
        split_dates=split_dates,
    )

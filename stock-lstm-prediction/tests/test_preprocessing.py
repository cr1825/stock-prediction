import numpy as np
import pytest

from src.data import generate_synthetic_data
from src.features import FEATURE_COLUMNS, add_technical_indicators
from src.preprocessing import create_sequences, prepare_data

LOOKBACK = 30


@pytest.fixture(scope="module")
def data():
    df = add_technical_indicators(generate_synthetic_data(800))
    return df, prepare_data(df, lookback=LOOKBACK, train_frac=0.7, val_frac=0.15)


def test_sequence_shapes(data):
    _, d = data
    n_feat = len(FEATURE_COLUMNS)
    for X, y in ((d.X_train, d.y_train), (d.X_val, d.y_val), (d.X_test, d.y_test)):
        assert X.ndim == 3 and X.shape[1:] == (LOOKBACK, n_feat)
        assert len(X) == len(y) > 0
    assert d.latest_window.shape == (1, LOOKBACK, n_feat)


def test_scaler_fitted_on_training_rows_only(data):
    df, d = data
    valid = df.dropna(subset=FEATURE_COLUMNS)
    n = len(valid) - 1  # last row has no target
    n_train = int(n * 0.7)
    expected = valid[FEATURE_COLUMNS].iloc[:n_train].mean().values
    np.testing.assert_allclose(d.scaler.mean_, expected, rtol=1e-6, atol=1e-9)
    # ... and differs from the mean over all data (proves no leakage from test)
    assert not np.allclose(d.scaler.mean_, valid[FEATURE_COLUMNS].mean().values)


def test_split_is_chronological(data):
    _, d = data
    s = d.split_dates
    assert s["train_end"] < s["val_start"] <= s["val_end"] < s["test_start"]


def test_target_is_next_day_return(data):
    df, d = data
    # actual_close / prev_close must equal exp(actual_return)
    ratio = np.log(d.test_meta["actual_close"] / d.test_meta["prev_close"])
    np.testing.assert_allclose(ratio.values, d.test_meta["actual_return"].values, atol=1e-9)
    # targets in scaled space invert back to the same returns
    back = d.target_scaler.inverse_transform(d.y_test.reshape(-1, 1)).ravel()
    np.testing.assert_allclose(back, d.test_meta["actual_return"].values, atol=1e-5)


def test_create_sequences_alignment():
    feats = np.arange(20, dtype=float).reshape(10, 2)
    targets = np.arange(10, dtype=float)
    X, y, idx = create_sequences(feats, targets, lookback=3)
    assert X.shape == (8, 3, 2)
    np.testing.assert_array_equal(X[0], feats[0:3])
    assert y[0] == targets[2] and idx[0] == 2


def test_too_little_data_raises():
    df = add_technical_indicators(generate_synthetic_data(120))
    with pytest.raises(ValueError):
        prepare_data(df, lookback=60)

import numpy as np
import pandas as pd

from src.data import generate_synthetic_data
from src.features import FEATURE_COLUMNS, add_technical_indicators, compute_rsi


def test_all_feature_columns_created_and_count():
    out = add_technical_indicators(generate_synthetic_data(300))
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    assert len(FEATURE_COLUMNS) >= 15


def test_no_nans_or_infs_after_warmup():
    out = add_technical_indicators(generate_synthetic_data(300))
    valid = out.dropna(subset=FEATURE_COLUMNS)
    assert len(valid) > 200
    assert np.isfinite(valid[FEATURE_COLUMNS].values).all()


def test_rsi_bounded_between_0_and_100():
    rsi = compute_rsi(generate_synthetic_data(500)["Close"]).dropna()
    assert rsi.between(0, 100).all()


def test_no_look_ahead_bias():
    """Features for past days must not change when future rows are added."""
    df = generate_synthetic_data(400)
    full = add_technical_indicators(df)
    truncated = add_technical_indicators(df.iloc[:-50])
    pd.testing.assert_frame_equal(
        full.loc[truncated.index, FEATURE_COLUMNS], truncated[FEATURE_COLUMNS]
    )


def test_index_without_volume_still_works():
    df = generate_synthetic_data(300)
    df["Volume"] = 0.0
    out = add_technical_indicators(df).dropna(subset=FEATURE_COLUMNS)
    assert len(out) > 200

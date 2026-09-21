import numpy as np
import pandas as pd

from src.evaluate import (backtest_long_flat, directional_accuracy,
                          evaluate_predictions, regression_metrics)


def test_perfect_prediction_has_zero_error():
    m = regression_metrics([100, 101, 102], [100, 101, 102])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["mape"] == 0


def test_mape_known_value():
    m = regression_metrics([100, 200], [110, 180])  # 10% and 10% errors
    assert abs(m["mape"] - 10.0) < 1e-9


def test_directional_accuracy():
    assert directional_accuracy([1, -1, 1, -1], [1, -1, -1, 1]) == 50.0


def _pred_df(pred_ret):
    prev = np.array([100.0, 101.0, 100.5, 102.0])
    act_ret = np.array([0.01, -0.005, 0.015, 0.002])
    return pd.DataFrame({
        "prev_close": prev,
        "actual_close": prev * np.exp(act_ret),
        "actual_return": act_ret,
        "predicted_return": pred_ret,
        "predicted_close": prev * np.exp(pred_ret),
    })


def test_evaluate_reports_baselines():
    res = evaluate_predictions(_pred_df(np.array([0.01, -0.005, 0.015, 0.002])))
    assert res["model"]["rmse"] < 1e-9
    assert res["rmse_improvement_vs_naive_pct"] > 99
    assert res["model"]["directional_accuracy"] == 100.0


def test_backtest_long_flat_avoids_down_days():
    stats, curve = backtest_long_flat(_pred_df(np.array([0.01, -0.005, 0.015, 0.002])))
    assert stats["strategy_total_return_pct"] > stats["buy_and_hold_total_return_pct"]
    assert len(curve) == 4

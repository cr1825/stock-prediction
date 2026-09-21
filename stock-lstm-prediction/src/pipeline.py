"""End-to-end pipeline: data -> features -> LSTM -> evaluation -> artifacts."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .data import fetch_stock_data, generate_synthetic_data
from .evaluate import backtest_long_flat, evaluate_predictions
from .features import add_technical_indicators
from .preprocessing import build_latest_window, prepare_data

logger = logging.getLogger(__name__)


def _inverse_return(scaled: np.ndarray, target_scaler) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()


def _next_business_day(last_date: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(last_date, periods=2)[1]  # ignores exchange holidays


def make_forecast(model, window: np.ndarray, target_scaler, last_close: float,
                  last_date: pd.Timestamp) -> Dict:
    pred_ret = float(_inverse_return(model.predict(window, verbose=0), target_scaler)[0])
    return {
        "as_of_date": str(last_date.date()),
        "forecast_date": str(_next_business_day(last_date).date()),
        "last_close": float(last_close),
        "predicted_return_pct": pred_ret * 100,
        "predicted_close": float(last_close * np.exp(pred_ret)),
        "direction": "UP" if pred_ret > 0 else "DOWN",
    }


def run_pipeline(cfg: Config, extra_callbacks: Sequence = (), save: bool = True,
                 make_plots: bool = True) -> Dict:
    """Run everything and return a dict with data, metrics and artifact paths."""
    # Imported here so that data/feature code works without TensorFlow installed.
    from tensorflow import keras

    from .model import build_lstm_model, train_model

    keras.utils.set_random_seed(cfg.seed)
    ticker = "SYNTHETIC" if cfg.synthetic else cfg.ticker.upper().strip()

    # 1. Data collection -------------------------------------------------------
    if cfg.synthetic:
        raw = generate_synthetic_data(seed=cfg.seed)
    else:
        raw = fetch_stock_data(ticker, cfg.start, cfg.end,
                               cache_dir=str(Path(cfg.data_dir) / "raw"))
    logger.info("Loaded %d daily rows (%s -> %s)", len(raw),
                raw.index[0].date(), raw.index[-1].date())

    # 2. Feature engineering + preprocessing ----------------------------------
    df_feat = add_technical_indicators(raw)
    data = prepare_data(df_feat, cfg.lookback, cfg.train_frac, cfg.val_frac)
    logger.info("Samples - train: %d, val: %d, test: %d", len(data.X_train),
                len(data.X_val), len(data.X_test))

    # 3. Model -----------------------------------------------------------------
    model = build_lstm_model(cfg.lookback, len(data.feature_columns),
                             cfg.lstm_units, cfg.dropout, cfg.learning_rate)
    history = train_model(model, data.X_train, data.y_train, data.X_val, data.y_val,
                          cfg.epochs, cfg.batch_size, cfg.verbose, extra_callbacks)

    # 4. Predict on the untouched test set -------------------------------------
    pred_ret = _inverse_return(model.predict(data.X_test, verbose=0), data.target_scaler)
    pred_df = data.test_meta.copy()
    pred_df["predicted_return"] = pred_ret
    pred_df["predicted_close"] = pred_df["prev_close"] * np.exp(pred_ret)

    metrics = evaluate_predictions(pred_df)
    backtest_stats, curve = backtest_long_flat(pred_df)
    metrics["backtest"] = backtest_stats
    metrics["split_dates"] = data.split_dates
    metrics["epochs_trained"] = len(history.history["loss"])

    # 5. Next-day forecast ------------------------------------------------------
    forecast = make_forecast(model, data.latest_window, data.target_scaler,
                             data.last_close, data.last_date)

    result = {
        "ticker": ticker, "config": cfg, "raw": raw, "df_features": df_feat,
        "pred_df": pred_df, "metrics": metrics, "history": history.history,
        "forecast": forecast, "backtest_curve": curve, "paths": {},
    }

    # 6. Save artifacts ----------------------------------------------------------
    if save:
        model_dir, out_dir = Path(cfg.model_dir), Path(cfg.output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{ticker}_lstm.keras"
        art_path = model_dir / f"{ticker}_artifacts.joblib"
        model.save(model_path)
        joblib.dump({"scaler": data.scaler, "target_scaler": data.target_scaler,
                     "feature_columns": data.feature_columns,
                     "lookback": cfg.lookback, "config": asdict(cfg)}, art_path)

        metrics_path = out_dir / f"{ticker}_metrics.json"
        metrics_path.write_text(json.dumps({**metrics, "forecast": forecast}, indent=2))
        pred_path = out_dir / f"{ticker}_test_predictions.csv"
        pred_df.to_csv(pred_path)
        result["paths"].update(model=str(model_path), artifacts=str(art_path),
                               metrics=str(metrics_path), predictions=str(pred_path))

    if make_plots:
        from . import visualize as viz

        out_dir = Path(cfg.output_dir)
        result["paths"].update(
            overview=viz.plot_price_overview(df_feat, ticker, out_dir / f"{ticker}_overview.png"),
            predictions_plot=viz.plot_predictions(pred_df, ticker, out_dir / f"{ticker}_predictions.png"),
            training=viz.plot_training_history(history.history, out_dir / f"{ticker}_training.png"),
            scatter=viz.plot_return_scatter(pred_df, metrics["model"]["directional_accuracy"],
                                            out_dir / f"{ticker}_return_scatter.png"),
            backtest=viz.plot_backtest(curve, ticker, out_dir / f"{ticker}_backtest.png"),
        )
    return result


def forecast_with_saved_model(ticker: str, model_dir: str = "models",
                              data_dir: str = "data") -> Dict:
    """Load a previously trained model and forecast the next trading day."""
    from tensorflow import keras

    ticker = ticker.upper().strip()
    model_path = Path(model_dir) / f"{ticker}_lstm.keras"
    art_path = Path(model_dir) / f"{ticker}_artifacts.joblib"
    if not model_path.exists() or not art_path.exists():
        raise FileNotFoundError(
            f"No trained model for {ticker} in '{model_dir}'. "
            f"Train one first: python main.py --ticker {ticker}"
        )
    model = keras.models.load_model(model_path)
    art = joblib.load(art_path)

    start = (pd.Timestamp.today() - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    raw = fetch_stock_data(ticker, start, cache_dir=str(Path(data_dir) / "raw"))
    df_feat = add_technical_indicators(raw)
    window = build_latest_window(df_feat, art["scaler"], art["feature_columns"], art["lookback"])
    valid = df_feat.dropna(subset=art["feature_columns"])
    return make_forecast(model, window, art["target_scaler"],
                         float(valid["Close"].iloc[-1]), valid.index[-1])

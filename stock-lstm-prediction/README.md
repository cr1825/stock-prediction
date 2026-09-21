# 📈 Stock Price Prediction with a Stacked LSTM

[!\[CI](https://github.com/cr1825/stock-lstm-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/stock-lstm-prediction/actions)
!\[Python](https://img.shields.io/badge/python-3.10--3.12-blue)
!\[License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine-learning project that downloads market data with **yfinance**, engineers
technical indicators, trains a **stacked LSTM** in TensorFlow/Keras to forecast the next
trading day, and evaluates it **honestly** against naive baselines. Includes a CLI, an
interactive **Streamlit dashboard**, unit tests and CI.

> ⚠️ \*\*Educational project - not financial advice.\*\* Markets are noisy and this model does not
> beat the market. See \[Limitations](#limitations).

## Highlights

* **Live data pipeline** - daily OHLCV from Yahoo Finance via `yfinance`, cached locally.
* **20 engineered features** - RSI, MACD (+signal/histogram), SMA/EMA distances, Bollinger
Bands, rolling volatility, momentum, volume ratios, all built without look-ahead bias.
* **Stacked LSTM** (2 LSTM layers + dropout + dense head), Huber loss, early stopping and
learning-rate scheduling.
* **Rigorous evaluation** - RMSE, MAE, MAPE and directional accuracy, always compared with a
*"tomorrow = today"* baseline and an *"always up"* baseline.
* **Data-integrity safeguards** - chronological split, scalers fitted on training data only,
automated tests that check for look-ahead bias and leakage.
* **Interactive dashboard** (Streamlit + Plotly) and a toy long/flat backtest.

## Project structure

```
stock-lstm-prediction/
├── main.py                 # CLI: train + evaluate + save plots/model
├── predict.py              # Next-day forecast from a saved model
├── app.py                  # Streamlit dashboard
├── src/
│   ├── config.py           # Hyper-parameters \& paths (dataclass)
│   ├── data.py             # yfinance download + caching + cleaning
│   ├── features.py         # Technical indicators (RSI, MACD, Bollinger, ...)
│   ├── preprocessing.py    # Target, chronological split, scaling, sequences
│   ├── model.py            # Stacked LSTM + training callbacks
│   ├── evaluate.py         # Metrics, baselines, toy backtest
│   ├── visualize.py        # Matplotlib charts
│   └── pipeline.py         # Orchestrates everything
├── tests/                  # pytest suite (runs offline)
├── outputs/                # Generated charts \& metrics (commit these!)
├── models/  data/          # Saved model / cached data (git-ignored)
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/cr1825/stock-lstm-prediction.git
cd stock-lstm-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 1\. Train \& evaluate (CLI)

```bash
python main.py --ticker AAPL
python main.py --ticker TSLA --start 2015-01-01 --epochs 80 --lookback 90
python main.py --help
```

This prints a results table and writes charts to `outputs/`, the model to `models/`.

### 2\. Interactive dashboard

```bash
streamlit run app.py
```

### 3\. Forecast with a saved model

```bash
python predict.py --ticker AAPL
```

### 4\. Run the tests (no internet needed)

```bash
pytest -q
```

## Methodology

**Target.** The model predicts the *next-day log return* and the price forecast is
`close\_t · exp(predicted\_return)`. Predicting returns instead of raw prices avoids the classic
trap where a model simply learns "tomorrow ≈ today" and still looks accurate.

**Features (20).** `log\_return`, `hl\_range`, `oc\_change`, `volume\_change`, `volume\_ratio`,
`rsi\_14`, `macd`, `macd\_signal`, `macd\_hist`, `close\_to\_sma\_{10,20,50}`, `close\_to\_ema\_{12,26}`,
`volatility\_{5,10,20}`, `bb\_width`, `bb\_position`, `momentum\_10`. Price-based features are
expressed as ratios so they are approximately stationary.

**Model.** `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16, ReLU) → Dense(1)`, trained with
Adam + Huber loss, early stopping (patience 10) and `ReduceLROnPlateau`. Input is a
60-day window of the 20 features.

**Split.** Chronological 70 % train / 15 % validation / 15 % test. The test set is never used
for training, early stopping or scaling.

**Leakage prevention.**

* `StandardScaler`s are fitted on training rows only.
* Features at day *t* use only data up to *t* (`tests/test\_features.py::test\_no\_look\_ahead\_bias`).
* The split is strictly time-ordered (`tests/test\_preprocessing.py`).

## Evaluation

|Metric|Meaning|
|-|-|
|**RMSE / MAE**|Price error in dollars|
|**MAPE**|Price error relative to the actual price|
|**Directional accuracy**|% of days the up/down call was right|
|**Naive baseline**|"Tomorrow's price = today's price" - the bar a model must beat|
|**Always-up baseline**|Directional accuracy of predicting "up" every day|
|**Long/flat backtest**|Toy strategy: hold when predicting up, else cash (no costs)|

### Results

> 🔧 \*\*Run the project yourself and paste your real numbers here.\*\* Example template:

|Ticker|Test period|RMSE (LSTM / naive)|MAPE (LSTM / naive)|Directional acc. (LSTM / always-up)|
|-|-|-|-|-|
|AAPL|`YYYY-MM-DD → YYYY-MM-DD`|`x.xx / x.xx`|`x.xx% / x.xx%`|`xx.x% / xx.x%`|

!\[Predictions](outputs/AAPL\_predictions.png)
!\[Indicators](outputs/AAPL\_overview.png)

<sub>(Images appear after you run `python main.py --ticker AAPL` and commit the `outputs/` folder.)</sub>

### Reading the prediction chart

One-day-ahead forecasts usually look like the actual price shifted by a day - the model leans
heavily on the latest close. That is normal and is exactly why the naive baseline is reported
next to every metric. A low MAPE alone does **not** mean the model is skilful; beating the
baseline and having directional accuracy meaningfully above the always-up rate does.

## Limitations

* Prices are close to a random walk; expect metrics near the naive baseline.
* Directional accuracy on a few hundred test days has a wide margin of error (±3-4 pts).
* The backtest ignores transaction costs, slippage and taxes, and a short test window is noisy.
* Only past prices/volume are used - no news, fundamentals or macro data.
* The next-business-day date ignores exchange holidays; if you run the model during market
hours the latest row may be an incomplete bar.
* Single-split evaluation; walk-forward validation would be more robust.

## Ideas for extending the project

* Walk-forward (expanding-window) cross-validation
* Compare against Auto-ARIMA and Prophet baselines
* GRU / Bidirectional LSTM / Transformer variants and hyper-parameter search (Keras Tuner / Optuna)
* Add market-wide features (S\&P 500, VIX) and sentiment from news headlines
* Multi-step forecasting and probabilistic (quantile) outputs
* Deploy the dashboard on Streamlit Community Cloud

## License

MIT - see [LICENSE](LICENSE).


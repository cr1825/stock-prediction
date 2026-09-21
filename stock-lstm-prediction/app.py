"""Interactive dashboard.  Run with:  streamlit run app.py"""

from datetime import date

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from tensorflow import keras

from src.config import Config
from src.features import compute_macd, compute_rsi
from src.pipeline import run_pipeline

st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")


class StreamlitProgress(keras.callbacks.Callback):
    """Feeds Keras epoch progress into a Streamlit progress bar."""

    def __init__(self, bar, total_epochs: int):
        super().__init__()
        self.bar, self.total = bar, total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.bar.progress(
            min((epoch + 1) / self.total, 1.0),
            text=f"Epoch {epoch + 1}/{self.total} - val_loss {logs.get('val_loss', 0):.4f}",
        )


# ------------------------------------------------------------------ figures ---
def fig_predictions(pred_df, ticker):
    fig = go.Figure()
    fig.add_scatter(x=pred_df.index, y=pred_df["actual_close"], name="Actual close",
                    line=dict(color="#1f77b4", width=2))
    fig.add_scatter(x=pred_df.index, y=pred_df["predicted_close"], name="LSTM prediction",
                    line=dict(color="#d62728", width=1.6, dash="dash"))
    fig.update_layout(title=f"{ticker}: one-day-ahead predictions on the test set",
                      hovermode="x unified", yaxis_title="Price", height=480,
                      legend=dict(orientation="h", y=1.08))
    return fig


def fig_overview(df, ticker, last_n=500):
    d = df.iloc[-last_n:].copy()
    close = df["Close"]
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    std = close.rolling(20).std()
    rsi = compute_rsi(close)
    macd, sig, hist = compute_macd(close)
    sl = d.index

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.2, 0.25])
    fig.add_scatter(x=sl, y=(sma20 + 2 * std).loc[sl], line=dict(width=0), showlegend=False,
                    hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=sl, y=(sma20 - 2 * std).loc[sl], fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(128,128,128,0.15)", name="Bollinger (20, 2σ)", row=1, col=1)
    fig.add_scatter(x=sl, y=d["Close"], name="Close", line=dict(color="#1f77b4"), row=1, col=1)
    fig.add_scatter(x=sl, y=sma20.loc[sl], name="SMA 20", line=dict(color="#ff7f0e", width=1), row=1, col=1)
    fig.add_scatter(x=sl, y=sma50.loc[sl], name="SMA 50", line=dict(color="#2ca02c", width=1), row=1, col=1)
    fig.add_scatter(x=sl, y=rsi.loc[sl], name="RSI (14)", line=dict(color="#9467bd"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_scatter(x=sl, y=macd.loc[sl], name="MACD", row=3, col=1)
    fig.add_scatter(x=sl, y=sig.loc[sl], name="Signal", row=3, col=1)
    fig.add_bar(x=sl, y=hist.loc[sl], name="Histogram", marker_color="grey", opacity=0.5, row=3, col=1)
    fig.update_layout(title=f"{ticker}: price & technical indicators", height=720,
                      hovermode="x unified", legend=dict(orientation="h", y=1.06))
    return fig


def fig_history(history):
    fig = go.Figure()
    fig.add_scatter(y=history["loss"], name="Train loss")
    fig.add_scatter(y=history["val_loss"], name="Validation loss")
    fig.update_layout(title="Training history", xaxis_title="Epoch", yaxis_title="Huber loss", height=380)
    return fig


def fig_backtest(curve):
    fig = go.Figure()
    fig.add_scatter(x=curve.index, y=curve["buy_and_hold"], name="Buy & hold", line=dict(color="grey"))
    fig.add_scatter(x=curve.index, y=curve["strategy"], name="LSTM long/flat", line=dict(color="#d62728"))
    fig.update_layout(title="Growth of $1 on the test period (no transaction costs)",
                      height=400, hovermode="x unified")
    return fig


# ------------------------------------------------------------------ sidebar ---
st.title("📈 LSTM Stock Price Predictor")
st.caption("Stacked-LSTM one-day-ahead forecasting on Yahoo Finance data. "
           "Educational project - **not financial advice**.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", "AAPL").upper().strip()
    start = st.date_input("Start date", date(2013, 1, 1), min_value=date(1990, 1, 1))
    lookback = st.slider("Look-back window (days)", 20, 120, 60, step=5)
    epochs = st.slider("Max epochs (early stopping enabled)", 5, 100, 30, step=5)
    units1 = st.select_slider("LSTM layer 1 units", [32, 64, 128], value=64)
    units2 = st.select_slider("LSTM layer 2 units", [16, 32, 64], value=32)
    with st.expander("Advanced"):
        synthetic = st.checkbox("Demo mode: synthetic random data (offline)", value=False)
    run = st.button("🚀 Fetch data & train model", type="primary", width="stretch")

if run:
    cfg = Config(ticker=ticker, start=str(start), lookback=lookback, epochs=epochs,
                 lstm_units=(units1, units2), verbose=0, synthetic=synthetic)
    bar = st.progress(0.0, text="Downloading data & building features...")
    try:
        # Nothing is written to disk from the dashboard; use main.py for saved artifacts.
        st.session_state["result"] = run_pipeline(
            cfg, extra_callbacks=[StreamlitProgress(bar, epochs)], save=False, make_plots=False
        )
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Could not complete the run: {exc}")
    finally:
        bar.empty()

res = st.session_state.get("result")
if res is None:
    st.info("Choose a ticker in the sidebar and click **Fetch data & train model**.")
    st.stop()

if res["config"].synthetic:
    st.warning("Demo mode: these results come from RANDOM synthetic data, not a real stock.")

m, f = res["metrics"], res["forecast"]
model_m, naive_m = m["model"], m["naive_persistence_baseline"]

st.subheader(f"{res['ticker']} - next trading day forecast")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"${f['last_close']:.2f}", help=f"As of {f['as_of_date']}")
c2.metric(f"Predicted close ({f['forecast_date']})", f"${f['predicted_close']:.2f}",
          f"{f['predicted_return_pct']:+.2f}%")
c3.metric("Predicted direction", f["direction"])
c4.metric("Test days", m["n_test_samples"])

st.subheader("Test-set evaluation (data the model never saw)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("RMSE", f"{model_m['rmse']:.3f}", f"naive: {naive_m['rmse']:.3f}", delta_color="off")
k2.metric("MAPE", f"{model_m['mape']:.2f}%", f"naive: {naive_m['mape']:.2f}%", delta_color="off")
k3.metric("Directional accuracy", f"{model_m['directional_accuracy']:.1f}%",
          f"always-up: {m['always_up_directional_accuracy']:.1f}%", delta_color="off")
k4.metric("RMSE vs naive baseline", f"{m['rmse_improvement_vs_naive_pct']:+.2f}%")
st.caption("A price model is only useful if it beats the naive baseline "
           "(\"tomorrow = today\") - compare the numbers above.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Predictions", "Price & indicators", "Training", "Backtest (toy)", "Data"])
with tab1:
    st.plotly_chart(fig_predictions(res["pred_df"], res["ticker"]), width="stretch")
with tab2:
    st.plotly_chart(fig_overview(res["raw"], res["ticker"]), width="stretch")
with tab3:
    st.plotly_chart(fig_history(res["history"]), width="stretch")
    sd = m["split_dates"]
    st.write(f"**Chronological split** - train: {sd['train_start']} → {sd['train_end']}, "
             f"validation: {sd['val_start']} → {sd['val_end']}, "
             f"test: {sd['test_start']} → {sd['test_end']}. "
             f"Epochs trained: {m['epochs_trained']}.")
with tab4:
    bt = m["backtest"]
    st.plotly_chart(fig_backtest(res["backtest_curve"]), width="stretch")
    st.write(f"Strategy: **{bt['strategy_total_return_pct']:+.1f}%** "
             f"(Sharpe {bt['strategy_sharpe']:.2f}) vs buy & hold: "
             f"**{bt['buy_and_hold_total_return_pct']:+.1f}%** "
             f"(Sharpe {bt['buy_and_hold_sharpe']:.2f}). "
             "Ignores costs and slippage; a short test window is very noisy.")
with tab5:
    st.dataframe(res["pred_df"].round(4).sort_index(ascending=False), width="stretch")
    st.download_button("Download test predictions (CSV)", res["pred_df"].to_csv().encode(),
                       file_name=f"{res['ticker']}_test_predictions.csv", mime="text/csv")

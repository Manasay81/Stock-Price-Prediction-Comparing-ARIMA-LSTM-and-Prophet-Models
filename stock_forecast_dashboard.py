import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go

st.set_page_config(layout="wide")

# Title
st.title("📈 Stock Forecasting Dashboard")

# Sidebar - Stock input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "LSTM", "Prophet", "Compare All"])

# Add Run Prediction Button in Sidebar
run_button = st.sidebar.button("Run Prediction")

# Run only if button clicked
if run_button:

    # Load Data
    df_raw = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    df_raw['MA_20'] = df_raw['Close'].rolling(20).mean()
    delta = df_raw['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_raw['RSI'] = 100 - (100 / (1 + rs))

    # Helper Functions
    def run_arima(df):
        train_size = int(len(df) * 0.8)
        train, test = df['Close'][:train_size], df['Close'][train_size:]
        model = pm.auto_arima(train, seasonal=False, suppress_warnings=True)
        preds = model.predict(n_periods=len(test))
        return preds, test.values, test.index

    def run_lstm(df):
        K.clear_session()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']])
        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        preds = model.predict(X_test)
        preds = scaler.inverse_transform(preds)
        y_test_inv = scaler.inverse_transform(y_test)
        test_index = df.index[seq_len + split:]
        return preds.flatten(), y_test_inv.flatten(), test_index

    def run_prophet(df):
        df = df.reset_index()[['Date', 'Close']].dropna()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
        train_size = int(0.8 * len(df))
        if train_size < 2:
            raise ValueError("Not enough data for Prophet after cleaning.")
        model = Prophet()
        model.fit(df.iloc[:train_size])
        future = model.make_future_dataframe(periods=len(df) - train_size)
        forecast = model.predict(future)
        preds = forecast['yhat'][train_size:].values
        actual = df['y'].iloc[train_size:].values
        dates = forecast['ds'][train_size:].values
        return preds, actual, dates

    def backtest_strategy(prices, preds):
        prices = np.array(prices).flatten()[-len(preds):]
        preds = np.array(preds).flatten()
        df_bt = pd.DataFrame({'Close': prices, 'Pred': preds})
        df_bt['Signal'] = np.where(df_bt['Pred'].shift(1) > df_bt['Close'].shift(1), 1, -1)
        df_bt['Returns'] = df_bt['Close'].pct_change()
        df_bt['Strategy_Returns'] = df_bt['Returns'] * df_bt['Signal']
        df_bt['Cumulative'] = (1 + df_bt['Strategy_Returns']).cumprod()
        sharpe = np.mean(df_bt['Strategy_Returns']) / np.std(df_bt['Strategy_Returns']) * np.sqrt(252)
        max_dd = (df_bt['Cumulative'] / df_bt['Cumulative'].cummax() - 1).min()
        return sharpe, max_dd, df_bt

    # Visualization per model
    if model_choice != "Compare All":
        if model_choice == "ARIMA":
            preds, true, dates = run_arima(df_raw)
            preds = np.ravel(preds)
            true = np.ravel(true)
            dates = pd.to_datetime(dates)
        elif model_choice == "LSTM":
            preds, true, dates = run_lstm(df_raw)
        else:
            preds, true, dates = run_prophet(df_raw)

        st.subheader(f"🔍 Actual vs {model_choice} Prediction")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=true, mode='lines', name='Actual', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines', name=f'{model_choice} Prediction', line=dict(dash='dash')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', yaxis=dict(autorange=True))
        st.plotly_chart(fig)

        st.subheader("📌 Evaluation Metrics")
        st.write(f"MAE: {mean_absolute_error(true, preds):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(true, preds)):.2f}")
        st.write(f"R²: {r2_score(true, preds):.2f}")

        # Buy/Sell Signals
        y_pred_array = np.array(preds)
        price_change = np.diff(y_pred_array, prepend=y_pred_array[0]) / y_pred_array
        signals = np.where(price_change > 0.01, 'BUY', 'SELL')
        df_signals = pd.DataFrame({'Date': dates, 'Signal': signals})
        st.subheader("💡 Buy/Sell Signal Suggestions")
        st.dataframe(df_signals.head(10))

        # Backtesting
        sharpe, max_dd, df_bt = backtest_strategy(df_raw['Close'].values, preds)
        st.subheader("📉 Backtesting Metrics")
        st.write(f"🔹 Sharpe Ratio: {sharpe:.2f}")
        st.write(f"🔹 Max Drawdown: {max_dd:.2%}")

    else:
        # Compare All
        preds_arima, true_arima, dates_arima = run_arima(df_raw)
        preds_lstm, true_lstm, dates_lstm = run_lstm(df_raw)
        preds_prophet, true_prophet, dates_prophet = run_prophet(df_raw)

        st.subheader("📊 RMSE Comparison")
        rmse_values = [np.sqrt(mean_squared_error(true_arima, preds_arima)),
                       np.sqrt(mean_squared_error(true_lstm, preds_lstm)),
                       np.sqrt(mean_squared_error(true_prophet, preds_prophet))]
        models = ['ARIMA', 'LSTM', 'Prophet']
        fig_rmse = go.Figure([go.Bar(x=models, y=rmse_values)])
        fig_rmse.update_layout(title='RMSE Comparison Across Models', yaxis_title='RMSE')
        st.plotly_chart(fig_rmse)

        # Boxplot of Errors
        st.subheader("📊 Prediction Errors Distribution")
        true_arima_flat = np.array(true_arima).flatten()
        preds_arima_flat = np.array(preds_arima).flatten()
        true_lstm_flat = np.array(true_lstm).flatten()
        preds_lstm_flat = np.array(preds_lstm).flatten()
        true_prophet_flat = np.array(true_prophet).flatten()
        preds_prophet_flat = np.array(preds_prophet).flatten()
        min_len = min(len(true_arima_flat), len(preds_lstm_flat), len(preds_prophet_flat))
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=true_arima_flat[-min_len:] - preds_arima_flat[-min_len:], name='ARIMA Error'))
        fig_box.add_trace(go.Box(y=true_lstm_flat[-min_len:] - preds_lstm_flat[-min_len:], name='LSTM Error'))
        fig_box.add_trace(go.Box(y=true_prophet_flat[-min_len:] - preds_prophet_flat[-min_len:], name='Prophet Error'))
        fig_box.update_layout(title='Distribution of Prediction Errors')
        st.plotly_chart(fig_box)

        # Cumulative Returns
        st.subheader("📊 Cumulative Strategy Returns")
        _, _, df_bt_arima = backtest_strategy(df_raw['Close'].values, preds_arima)
        _, _, df_bt_lstm = backtest_strategy(df_raw['Close'].values, preds_lstm)
        _, _, df_bt_prophet = backtest_strategy(df_raw['Close'].values, preds_prophet)
        fig_cum = go.Figure()

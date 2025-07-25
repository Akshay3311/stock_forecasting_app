import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import datetime

# Forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_excel('AAPL_preprocessed.xls')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Fix index frequency for SARIMA/ARIMA plotting
df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp()

# UI
st.title("📈 Stock Price Forecasting")
st.write("Select a forecasting model to view the predicted stock prices for the next 30 days.")

model_option = st.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])

forecast_period = 30  # days
actual = df['Close'][-60:]

# ---------------------- ARIMA ----------------------
if model_option == "ARIMA":
    model = ARIMA(df['Close'], order=(5, 1, 0))
    results = model.fit()
    forecast = results.forecast(steps=forecast_period)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)

    forecast_series = pd.Series(forecast.values, index=future_dates, name="ARIMA Forecast")

    st.subheader("ARIMA Forecast")
    plt.figure(figsize=(12, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(forecast_series.index, forecast_series.values, label="ARIMA Forecast", color='orange')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# ---------------------- SARIMA ----------------------
elif model_option == "SARIMA":
    model = SARIMAX(df['Close'], order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))  # Try 30 for monthly seasonality
    results = model.fit(disp=False)
    forecast = results.forecast(steps=forecast_period)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)

    forecast_series = pd.Series(forecast.values, index=future_dates, name="SARIMA Forecast")

    st.subheader("SARIMA Forecast")
    plt.figure(figsize=(12, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(forecast_series.index, forecast_series.values, label="SARIMA Forecast", color='green')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# ---------------------- Prophet ----------------------
elif model_option == "Prophet":
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)

    st.subheader("Prophet Forecast")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

# ---------------------- LSTM ----------------------
elif model_option == "LSTM":
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['Close']])

    X_test = []
    n_lookback = 60
    for i in range(n_lookback, len(data_scaled)):
        X_test.append(data_scaled[i - n_lookback:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = load_model('lstm_model.h5')
    prediction_scaled = model.predict(X_test[-forecast_period:])
    prediction = scaler.inverse_transform(prediction_scaled)

    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_period + 1)]
    forecast_series = pd.Series(prediction.flatten(), index=future_dates, name="LSTM Forecast")

    st.subheader("LSTM Forecast")
    plt.figure(figsize=(12, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    plt.plot(forecast_series.index, forecast_series.values, label="LSTM Forecast", color='blue')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
# Trigger rebuild on Streamlit

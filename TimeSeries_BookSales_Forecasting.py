import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler

# Import and combine CSVs (placeholder names)
files = ['dataset_1.csv', 'dataset_2.csv']
data = [pd.read_csv(file, encoding='latin-1') for file in files]
combined = pd.concat(data, ignore_index=True)

# Clean and prepare data
combined.fillna(0, inplace=True)
combined['Book_Metadata_Field'] = combined['Book_Metadata_Field'].astype(str)
combined['End Date'] = pd.to_datetime(combined['End Date'], errors='coerce', dayfirst=True)
combined.set_index('End Date', inplace=True)

# Resample weekly sales data
trend_resampled = combined.groupby('ISBN').resample('W').sum().reset_index()
trend_resampled.fillna(0, inplace=True)

# Filter books of interest (anonymised)
book_a = trend_resampled[trend_resampled['Title'].str.contains("Book A", case=False, na=False)].copy()
book_b = trend_resampled[trend_resampled['Title'].str.contains("Book B", case=False, na=False)].copy()
book_a = book_a[book_a['End Date'] >= "2012-01-01"]
book_b = book_b[book_b['End Date'] >= "2012-01-01"]
book_a.set_index('End Date', inplace=True)
book_b.set_index('End Date', inplace=True)

# Stationarity test
def check_stationarity(series, label):
    result = adfuller(series.dropna())
    print(f"{label} - p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Series is stationary.")
    else:
        print("Series is NOT stationary.")

check_stationarity(book_a['Value'], "Book A")
check_stationarity(book_b['Value'], "Book B")

# Decomposition
seasonal_period = 52
book_a_decomp = seasonal_decompose(book_a['Value'], model='additive', period=seasonal_period)
book_b_decomp = seasonal_decompose(book_b['Value'], model='additive', period=seasonal_period)
book_a_decomp.plot()
plt.suptitle("Decomposition - Book A")
plt.show()

book_b_decomp.plot()
plt.suptitle("Decomposition - Book B")
plt.show()

# Forecast horizon
forecast_horizon = 32

# Auto ARIMA
def run_auto_arima(series, label, seasonal_period=26):
    model = auto_arima(
        series[:-forecast_horizon],
        seasonal=True, m=seasonal_period,
        stepwise=True, suppress_warnings=True
    )
    print(f"Auto ARIMA summary for {label}")
    print(model.summary())
    forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

    plt.figure(figsize=(12, 6))
    plt.plot(series.index[:-forecast_horizon], series[:-forecast_horizon], label='Training')
    plt.plot(series.index[-forecast_horizon:], series[-forecast_horizon:], label='Actual')
    plt.plot(series.index[-forecast_horizon:], forecast, label='Forecast', color='green')
    plt.fill_between(series.index[-forecast_horizon:], conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3)
    plt.title(f"Forecast - {label}")
    plt.legend()
    plt.grid()
    plt.show()

    return model, forecast

arima_model_a, forecast_a = run_auto_arima(book_a['Value'], "Book A")
arima_model_b, forecast_b = run_auto_arima(book_b['Value'], "Book B")

# Simple LSTM implementation
def create_input_sequences(lookback, forecast, sequence_data):
    X, y = [], []
    for i in range(lookback, len(sequence_data) - forecast + 1):
        X.append(sequence_data[i - lookback:i])
        y.append(sequence_data[i:i + forecast])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

lookback = 12

def run_lstm(series, label):
    series = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = create_input_sequences(lookback, 1, scaled)
    model = build_lstm_model((lookback, 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    last_seq = scaled[-lookback:].reshape(1, lookback, 1)
    forecast = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_seq)
        forecast.append(pred[0][0])
        last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    print(f"{label} - Forecast (LSTM):")
    print(forecast.flatten())
    return forecast.flatten()

forecast_lstm_a = run_lstm(book_a['Value'], "Book A")
forecast_lstm_b = run_lstm(book_b['Value'], "Book B")

# Evaluate LSTM forecasts
def evaluate_forecast(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"{label} - MAE: {mae:.2f}, MAPE: {mape:.2f}")

evaluate_forecast(book_a['Value'][-forecast_horizon:], forecast_lstm_a, "Book A (LSTM)")
evaluate_forecast(book_b['Value'][-forecast_horizon:], forecast_lstm_b, "Book B (LSTM)")

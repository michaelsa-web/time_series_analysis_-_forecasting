# time_series_analysis_-_forecasting
This project applies classical and machine learning time series techniques—including ARIMA, XGBoost, LSTM, and hybrid models—to forecast future book sales based on historical weekly sales data. The solution identifies seasonal trends and sales patterns to support data-driven inventory and stock planning decisions.

# Time Series Book Sales Forecasting

This project uses time series analysis to forecast future book sales for two anonymised titles based on historical weekly sales data. It explores statistical models (ARIMA), machine learning (XGBoost), deep learning (LSTM), and hybrid forecasting methods to identify patterns and support data-driven inventory planning.

## 📊 Project Overview

- **Data:** Weekly aggregated sales for two anonymised books (2012–2024)
- **Objective:** Forecast the next 32 weeks of sales to inform stock control and investment strategies
- **Models Used:**
  - Classical: Auto ARIMA, Seasonal Decomposition, ADF, ACF/PACF
  - Machine Learning: XGBoost pipeline with deseasonalisation and detrending
  - Deep Learning: LSTM with Keras Tuner
  - Hybrid Models: Sequential (SARIMA + LSTM residuals) and Parallel (SARIMA & LSTM weighted)

## 🧠 Key Skills Demonstrated

- Time series preprocessing, stationarity tests, and decomposition
- Forecast model tuning (grid search, hyperparameter optimisation)
- Evaluation with MAE and MAPE
- LSTM-based sequence modelling and hybrid architecture integration
- Visualisation and comparison of forecasts across multiple time horizons (weekly and monthly)

## 📁 Files

- `TimeSeries_BookSales_Forecasting.py` — Full source code with model pipelines
- `README.md` — This file

## 🚀 How to Run

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt

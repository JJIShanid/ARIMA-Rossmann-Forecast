# ARIMA-Rossmann-Forecast
# ARIMA Time Series Forecasting — Rossmann Store Sales

## Overview

This project demonstrates a complete workflow for **time series forecasting using ARIMA/SARIMAX models** on the Rossmann store sales dataset. The aim is to provide **accurate daily sales forecasts** for individual stores, along with clear diagnostics, metrics, and visualizations suitable for both academic evaluation and business decision-making.

---



markdown
Copy code

---

## Dataset

- **Source:** Rossmann Store Sales dataset (publicly available)
- **Columns used:**  
  - `Date`: daily date  
  - `Store`: store number  
  - `Sales`: daily sales  
- **Preprocessing:**  
  - Filtered by selected store  
  - Resampled to daily sales (`resample('D')`)  
  - Missing values filled with 0  

---

## Workflow / Methodology

The project follows a **robust ARIMA pipeline**:

1. **Exploratory Analysis & Checks**
   - Display basic stats and plot sales trends
   - Verify columns and data integrity

2. **Stationarity Check**
   - ADF (Augmented Dickey-Fuller) test
   - p-value ≤ 0.05 indicates stationarity

3. **Autocorrelation Analysis**
   - ACF (Auto-correlation Function)
   - PACF (Partial Auto-correlation Function)

4. **ARIMA/SARIMAX Modeling**
   - Optional **automatic grid search** for (p,d,q) using **AIC**
   - Include optional weekly seasonality
   - Fit final model on full series

5. **Train/Test Forecast Evaluation**
   - Split: last `h` days as test, rest as training
   - Forecast comparison: actual vs predicted
   - Metrics:
     - **MAE** (Mean Absolute Error)  
     - **RMSE** (Root Mean Squared Error)  
     - **MAPE** (Mean Absolute Percentage Error) excluding zeros to avoid infinity  

6. **Diagnostics**
   - Residual plot
   - Residual density
   - Ljung-Box test for autocorrelation in residuals

7. **Forecast Export**
   - Save CSV for forecast and confidence intervals
   - Ready for business interpretation

---

## Streamlit Dashboard — How to Interact

The dashboard (`app.py`) provides **interactive forecasting**. Follow these steps:

1. **Start the app**:
```bash
python -m streamlit run app.py
Upload dataset:

Use the sidebar to upload train.csv or leave blank to use the default path.

Ensure the CSV has Date, Store, and Sales columns.

Select store:

Use the dropdown in the sidebar to select the store number you want to forecast.

Set forecast horizon:

Adjust the slider to select how many days ahead you want to forecast (30–180 days).

Choose ARIMA parameters:

Auto-fit: Small grid search will automatically select the best (p,d,q) based on AIC.

Manual: Enter your own (p,d,q) values if desired.

Run ARIMA:

Click the Run ARIMA button to fit the model and generate forecasts.

The app will show:

Selected or manual ARIMA order

Model summary (click expander to view full statsmodels summary)

Forecast metrics: MAE, RMSE, MAPE (handles zero actuals)

Visualizations:

Forecast vs Actual: Interactive Plotly line chart with confidence intervals

ACF/PACF plots: Optional check in sidebar to view autocorrelations

Residuals: Plots and density estimates

Ljung-Box test: Displays p-value for residual autocorrelation

Download forecast:

After the model runs, download the forecast CSV using the Download Forecast CSV button.

Key Results (Example)
Forecast horizon: 90 days

Selected store: 1

ARIMA order (auto-selected): (1,1,1)

Train/Test Metrics:

MAE: 455.21

RMSE: 580.34

MAPE: 12.5% (excluding zeros)

Insights for Business:

Weekly spikes observed on Fridays

ARIMA predicts minor sales increase in the next 3 months

Confidence intervals provide range for risk-aware planning

Files Produced
outputs/sarimax_store{X}_model.pkl — saved SARIMAX model

outputs/forecast_store{X}_arima.csv — predicted sales with confidence intervals

Interactive dashboard via Streamlit

Notebook showing step-by-step analysis and plots

How to Use
Clone the repository

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the notebook to generate forecasts

Run the Streamlit app for interactive analysis and download

Author

Ishan — MSc Data Science — Business Analytics, SRH University, Fürth, Germany


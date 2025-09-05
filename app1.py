
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import io
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIMA Time Series Dashboard", layout="wide")
st.title(" ARIMA Time Series Forecasting Dashboard (Rossmann)")

# --- Data upload / load ---
st.sidebar.header("Data")
upload = st.sidebar.file_uploader("Upload train.csv (Rossmann) or leave blank to use default path", type=['csv'])
if upload:
    df = pd.read_csv(upload, parse_dates=['Date'])
else:
    DATA_PATH = r"C:\Users\asus\Desktop\Masters Project & Thesis\Master Projects\time series project rossman\train.csv"
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    except FileNotFoundError:
        st.error(f"train.csv not found. Please upload file in sidebar.")
        st.stop()

# sanitize columns
if 'Sales' in df.columns:
    df.rename(columns={'Sales':'sales'}, inplace=True)
df = df[['Date','Store','sales']].dropna()
df['Date'] = pd.to_datetime(df['Date'])

# --- Sidebar options ---TT
stores = sorted(df['Store'].unique())
store = st.sidebar.selectbox("Select store", stores)
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=30, max_value=180, value=90)
auto_fit = st.sidebar.checkbox("Auto-select (p,d,q) small grid by AIC", value=True)
p_input = st.sidebar.number_input("p", min_value=0, max_value=5, value=1)
d_input = st.sidebar.number_input("d", min_value=0, max_value=2, value=1)
q_input = st.sidebar.number_input("q", min_value=0, max_value=5, value=1)
run_model = st.sidebar.button("Run ARIMA")

# --- Prepare series ---TT
store_df = df[df['Store']==store].set_index('Date').resample('D')['sales'].sum().fillna(0)
ts = store_df

st.markdown(f"### Store {store} — data period: {ts.index.min().date()} to {ts.index.max().date()} (n={len(ts)})")

# show ADF test
def adf_summary(series):
    res = adfuller(series)
    return res[0], res[1]

adf_stat, adf_p = adf_summary(ts)
st.write(f"ADF stat: {adf_stat:.4f}, p-value: {adf_p:.4f}  (p<=0.05 suggests stationarity)")

# run model 
if run_model:
    # determine order
    if auto_fit:
        import itertools
        p = q = range(0,3)
        d = [0,1]
        best_aic = np.inf
        best_order = (1,1,1)
        for comb in itertools.product(p,d,q):
            try:
                m = SARIMAX(ts, order=comb, enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                if r.aic < best_aic:
                    best_aic = r.aic
                    best_order = comb
            except:
                continue
        order = best_order
        st.write(f"Selected order by AIC: {order} (AIC={best_aic:.2f})")
    else:
        order = (p_input, d_input, q_input)
        st.write(f"Using manual order: {order}")

    # fit on train (leave last horizon as test)
    train = ts[:-horizon]
    test = ts[-horizon:]
    model = SARIMAX(train, order=order, seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # model summary as text
    with st.expander("Model summary (statsmodels)"):
        st.text(res.summary().as_text())

    # forecast
    fc = res.get_forecast(steps=horizon)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int()

    # metrics: exclude zero actuals from MAPE
    mask = test > 0
    mae = mean_absolute_error(test, fc_mean)
    rmse = math.sqrt(mean_squared_error(test, fc_mean))
    if mask.sum() > 0:
        mape = np.mean(np.abs((test[mask] - fc_mean[mask]) / test[mask])) * 100
    else:
        mape = np.nan

    st.metric("MAE", f"{mae:,.2f}")
    st.metric("RMSE", f"{rmse:,.2f}")
    st.metric("MAPE (%)", f"{mape:.2f}" if not np.isnan(mape) else "N/A (zeros present)")

    # plot interactive forecast vs actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, mode='lines+markers', name='Forecast', line=dict(color='orange')))
    if not fc_ci.empty:
        fig.add_trace(go.Scatter(x=fc_ci.index, y=fc_ci.iloc[:,0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc_ci.index, y=fc_ci.iloc[:,1], fill='tonexty', mode='lines', line=dict(width=0), name='CI', fillcolor='rgba(255,165,0,0.2)'))
    fig.update_layout(title=f"Store {store} Forecast vs Actual (h={horizon})", xaxis_title="Date", yaxis_title="Sales",
                      template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ACF/PACF plots
    if st.sidebar.checkbox("Show ACF/PACF"):
        col1, col2 = st.columns(2)
        with col1:
            fig_acf, ax = plt.subplots(figsize=(6,3))
            plot_acf(train, lags=40, ax=ax)
            st.pyplot(fig_acf)
        with col2:
            fig_pacf, ax = plt.subplots(figsize=(6,3))
            plot_pacf(train, lags=40, ax=ax)
            st.pyplot(fig_pacf)

    # Residuals and tests
    resid = res.resid
    col3, col4 = st.columns(2)
    with col3:
        fig_r, ax = plt.subplots(figsize=(6,3))
        ax.plot(resid); ax.set_title('Residuals')
        st.pyplot(fig_r)
    with col4:
        fig_k, ax = plt.subplots(figsize=(6,3))
        resid.plot(kind='kde', ax=ax); ax.set_title('Residual density')
        st.pyplot(fig_k)

    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    st.write("Ljung-Box p-value (lag10):", lb['lb_pvalue'].values[0])

    # download forecast (fitted on full series now)
    full_model = SARIMAX(ts, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fut = full_model.get_forecast(steps=horizon)
    fut_mean = fut.predicted_mean
    fut_ci = fut.conf_int()
    fut_df = pd.DataFrame({
        'date': fut_mean.index,
        'forecast': fut_mean.values,
        'lower_ci': fut_ci.iloc[:,0].values,
        'upper_ci': fut_ci.iloc[:,1].values
    })
    st.download_button("Download Forecast CSV", data=fut_df.to_csv(index=False), file_name=f"forecast_store{store}.csv", mime='text/csv')

#Experiment 15: Build a Weather Data Visualization App (Temperature Trends Over Time)
# weather_trends_streamlit_app.py
# Streamlit app: Weather Data Visualization & Analysis
# Features:
# - Load CSV or use demo data
# - Parse dates, handle multiple cities
# - Resample (daily/weekly/monthly), rolling means
# - Time-series decomposition (trend/seasonality/residual)
# - Compare multiple cities with interactive plots (plotly)
# - Correlation heatmap, anomaly detection (z-score)
# - Download filtered data
# - Attractive UI with custom CSS, responsive layout

import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

# Optional: seasonal_decompose (statsmodels) for decomposition
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False

st.set_page_config(page_title="Weather Trends Explorer", layout="wide", initial_sidebar_state="expanded")

# --- Styling ---
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #071029 100%); color: #e6eef8; }
    .main .block-container { padding-top:1.5rem; padding-bottom:2.0rem; }
    .stButton>button { background: linear-gradient(90deg,#06b6d4,#7c3aed); color: white; border: none; }
    .big-title { font-size:30px; font-weight:700; color: #e6eef8; }
    .muted { color: #9fb0c8; }
    .card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper functions ---

@st.cache_data
def generate_demo_data():
    rng = pd.date_range(start="2023-01-01", periods=730, freq="D")
    cities = ["Mumbai", "Delhi", "Bengaluru", "Kolkata"]
    rows = []
    np.random.seed(42)
    for city in cities:
        base = {
            "Mumbai": 28,
            "Delhi": 22,
            "Bengaluru": 24,
            "Kolkata": 27,
        }.get(city, 25)
        seasonal = 6 * np.sin(2 * np.pi * (rng.dayofyear / 365.25))
        noise = np.random.normal(scale=2.5, size=len(rng))
        tavg = base + seasonal + noise + (np.linspace(0, 0.6, len(rng)))
        tmin = tavg - np.random.uniform(3, 7, size=len(rng))
        tmax = tavg + np.random.uniform(3, 7, size=len(rng))
        for d, a, mi, ma in zip(rng, tavg, tmin, tmax):
            rows.append({"date": d, "city": city, "tmin": round(mi, 2), "tmax": round(ma, 2), "tavg": round(a, 2)})
    df = pd.DataFrame(rows)
    return df


def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
    return df


def prepare_df(df, date_col='date'):
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])
    # Ensure tavg exists
    if 'tavg' not in df.columns:
        if {'tmin', 'tmax'}.issubset(df.columns):
            df['tavg'] = (df['tmin'] + df['tmax']) / 2.0
        else:
            raise ValueError("Data must contain 'tavg' or both 'tmin' and 'tmax'")
    # Basic cleaning
    if 'city' not in df.columns:
        df['city'] = 'Unknown'
    return df


def resample_and_aggregate(df, freq='M'):
    # freq: 'D', 'W', 'M'
    df = df.set_index('date')
    agg = df.groupby('city').resample(freq).agg({'tmin':'mean','tmax':'mean','tavg':'mean'}).reset_index()
    return agg


def add_rolling(df, window_days=7, how='D'):
    df = df.sort_values('date')
    df['tavg_roll'] = df.groupby('city')['tavg'].transform(lambda s: s.rolling(window=window_days, min_periods=1).mean())
    return df


def compute_anomalies(df, z_thresh=2.5):
    df = df.copy()
    df['zscore'] = df.groupby('city')['tavg'].transform(lambda s: stats.zscore(s.fillna(s.mean())))
    df['anomaly'] = df['zscore'].abs() > z_thresh
    return df


def seasonal_decompose_safe(series, period=None, model='additive'):
    if not STATSmodels_AVAILABLE:
        return None
    # statsmodels requires an index with freq; attempt to infer
    try:
        result = seasonal_decompose(series.dropna(), period=period, model=model, extrapolate_trend='freq')
        return result
    except Exception:
        return None


def forecast_linear(df, city, periods=30):
    # Simple linear forecast
    sub = df[df['city'] == city].sort_values('date')
    if len(sub) < 10:
        return None
    X = np.arange(len(sub)).reshape(-1, 1)
    y = sub['tavg'].values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(sub), len(sub) + periods).reshape(-1, 1)
    future_y = model.predict(future_X)
    future_dates = pd.date_range(sub['date'].max() + pd.Timedelta(days=1), periods=periods, freq='D')
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_y})
    return forecast_df


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    return buffer.getvalue()


# --- Helper for decomposition plotting ---
def make_decomp_plotly(decomp, city_name):
    """Create plotly figure for time series decomposition"""
    obs = decomp.observed
    trend = decomp.trend
    seasonal = decomp.seasonal
    resid = decomp.resid
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=obs.index, y=obs.values, name='Observed', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name='Trend', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal.values, name='Seasonal', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, name='Residual', line=dict(color='orange')), row=4, col=1)
    
    fig.update_layout(
        title=f'{city_name} - Time Series Decomposition',
        template='plotly_dark',
        height=800,
        showlegend=False
    )
    return fig


def forecast_linear(df, city, periods=30):
    """Simple linear forecast using sklearn"""
    try:
        from sklearn.linear_model import LinearRegression
        sub = df[df['city'] == city].sort_values('date')
        if len(sub) < 10:
            return None
        X = np.arange(len(sub)).reshape(-1, 1)
        y = sub['tavg'].values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(sub), len(sub) + periods).reshape(-1, 1)
        future_y = model.predict(future_X)
        future_dates = pd.date_range(sub['date'].max() + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_y})
        return forecast_df
    except ImportError:
        return None


def to_excel_bytes(df):
    """Convert dataframe to Excel bytes for download"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    return buffer.getvalue()


# --- Layout ---
st.markdown("<div class='big-title'>🌤️ Weather Trends Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Interactive time-series analysis: smoothing, decomposition, comparison, anomalies, and export.</div>", unsafe_allow_html=True)
st.write("---")

# Sidebar controls
with st.sidebar:
    st.markdown("### 🛠️ Data & Controls")
    data_source = st.radio("📊 Data source", options=["Demo data", "Upload CSV"], index=0)
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("📁 Upload CSV file (columns: date, city, tavg or tmin/tmax)", type=['csv','txt'])
        if uploaded_file is None:
            st.info("Upload a CSV or switch to Demo data to see the app in action.")
    
    st.markdown("---")
    date_col = st.text_input("📅 Date column name", value='date')
    city_col = st.text_input("🏙️ City column name", value='city')
    
    st.markdown("---")
    agg_freq = st.selectbox("📈 Aggregation", options=[('D','Daily'),('W','Weekly'),('M','Monthly')], format_func=lambda x: x[1])
    agg_freq_val = agg_freq[0]
    rolling_window = st.slider("🔄 Rolling window (periods)", 1, 90, 14)
    z_thresh = st.slider("⚠️ Anomaly z-score threshold", 1.5, 4.5, 2.8)
    show_decomp = st.checkbox("📊 Show time-series decomposition", value=True)
    
    st.markdown("---")
    st.caption("💡 Tip: Use the Download buttons to export filtered data.")

# Load data
if data_source == 'Demo data':
    df = generate_demo_data()
else:
    if uploaded_file is None:
        st.stop()
    df = load_csv(uploaded_file)

try:
    df = prepare_df(df, date_col=date_col)
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

# Controls depending on data
cities = sorted(df['city'].dropna().unique())
selected_cities = st.multiselect("🏙️ Select cities to compare", options=cities, default=cities[:2] if len(cities)>=2 else cities)
if not selected_cities:
    st.warning("⚠️ Select at least one city to display charts.")
    st.stop()

# Date range
min_date = df[date_col].min().date()
max_date = df[date_col].max().date()
start_date, end_date = st.date_input("📅 Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter
mask = (df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date)) & (df['city'].isin(selected_cities))
filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("⚠️ No data in the selected date range / cities.")
    st.stop()

# Resample
try:
    agg = resample_and_aggregate(filtered, freq=agg_freq_val)
    agg = add_rolling(agg, window_days=rolling_window)
    agg = compute_anomalies(agg, z_thresh=z_thresh)
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# Top row: summary stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Overview")
    st.markdown(f"**Selected cities:** {', '.join(selected_cities)}  <br> **Date range:** {start_date} — {end_date}", unsafe_allow_html=True)
    st.markdown("</div>")
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric(label="📈 Records (filtered)", value=len(filtered))
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    mean_t = round(filtered['tavg'].mean(),2)
    st.metric(label="🌡️ Mean Tavg (°C)", value=f"{mean_t}")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    std_t = round(filtered['tavg'].std(),2)
    st.metric(label="📉 Std Dev (°C)", value=f"{std_t}")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# Main charts
st.subheader("🌡️ Temperature Trends & Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🔍 Decomposition", "⚠️ Anomalies", "📊 Correlation", "🔮 Forecast"])

with tab1:
    # Multi-city comparison line plot (agg)
    fig = go.Figure()
    for c in selected_cities:
        sub = agg[agg['city']==c]
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['tavg'], mode='lines+markers', name=f"{c} (avg)"))
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['tavg_roll'], mode='lines', name=f"{c} (roll {rolling_window})", line={'dash':'dash'}))

    fig.update_layout(legend={'orientation':'h','yanchor':'bottom','y':1.02,'xanchor':'right','x':1},
                      margin={'t':30,'b':10,'l':20,'r':20},
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Additional: Boxplot for seasonality
    st.markdown("**Seasonal Boxplot (Monthly)**")
    agg['month'] = agg['date'].dt.month
    fig_box = px.box(agg, x='month', y='tavg', color='city', title='Monthly Temperature Distribution')
    fig_box.update_layout(template='plotly_dark')
    st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.markdown("#### 🔍 Time Series Decomposition")
    for city in selected_cities:
        st.markdown(f"**Decomposition for {city}**")
        sub = agg[agg['city']==city].set_index('date')
        series = sub['tavg']
        period = None
        if agg_freq_val == 'M':
            period = 12
        elif agg_freq_val == 'W':
            period = 52
        else:
            period = 365 if len(series) >= 365 else None
        
        if STATSmodels_AVAILABLE and period and len(series) >= 2 * period:
            decomp = seasonal_decompose_safe(series, period=period)
            if decomp is not None:
                dc_fig = make_decomp_plotly(decomp, city)
                st.plotly_chart(dc_fig, use_container_width=True)
            else:
                st.info(f"❌ Could not decompose time series for {city}. Try using monthly aggregation with more data.")
        else:
            st.info(f"📊 Time-series decomposition requires statsmodels and sufficient data (at least {2 * period if period else 'N/A'} periods). Current data: {len(series)} periods.")

with tab3:
    st.markdown("#### ⚠️ Anomaly Detection")
    for city in selected_cities:
        st.markdown(f"**Anomalies for {city}**")
        anomalies = agg[(agg['city']==city) & (agg['anomaly'])]
        if not anomalies.empty:
            st.markdown(f"🔍 Found {len(anomalies)} anomalies (z-score > {z_thresh})")
            st.dataframe(anomalies[['date','tavg','zscore']].sort_values('date'), use_container_width=True)
            # Plot anomalies
            fig_anom = go.Figure()
            sub = agg[agg['city']==city]
            fig_anom.add_trace(go.Scatter(x=sub['date'], y=sub['tavg'], mode='lines', name='Temperature', line=dict(color='lightblue')))
            fig_anom.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['tavg'], mode='markers', name='Anomalies', marker=dict(color='red', size=10, symbol='x')))
            fig_anom.update_layout(title=f"🌡️ Temperature Anomalies - {city}", template='plotly_dark')
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.success(f"✅ No anomalies detected for {city} with threshold z > {z_thresh}.")
    
    if len(selected_cities) > 1:
        st.markdown("---")
        st.markdown("**📊 Anomaly Summary Across Cities**")
        anomaly_summary = agg.groupby('city')['anomaly'].sum().reset_index()
        fig_bar = px.bar(anomaly_summary, x='city', y='anomaly', title='Total Anomalies by City')
        fig_bar.update_layout(template='plotly_dark')
        st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.markdown("#### 📊 Correlation & Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔗 Inter-city correlation (tavg)**")
        pivot = agg.pivot_table(index='date', columns='city', values='tavg')
        corr = pivot.corr()
        figc = px.imshow(corr, text_auto=True, aspect='auto', title='Temperature Correlation Matrix', 
                        labels=dict(x='City', y='City', color='Correlation'))
        figc.update_layout(template='plotly_dark')
        st.plotly_chart(figc, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Temperature Distribution**")
        fig_hist = px.histogram(agg, x='tavg', color='city', nbins=30, title='Temperature Distribution by City',
                               labels={'tavg': 'Average Temperature (°C)', 'count': 'Frequency'})
        fig_hist.update_layout(template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Additional statistics
    st.markdown("---")
    st.markdown("**📋 Statistical Summary by City**")
    stats_summary = agg.groupby('city')['tavg'].agg(['mean', 'std', 'min', 'max']).round(2)
    st.dataframe(stats_summary, use_container_width=True)

with tab5:
    st.markdown("#### 🔮 Temperature Forecasting")
    forecast_periods = st.slider("📅 Forecast periods (days)", 7, 365, 30)
    
    for city in selected_cities:
        st.markdown(f"**Linear Forecast for {city}**")
        forecast_df = forecast_linear(agg, city, periods=forecast_periods)
        if forecast_df is None:
            st.warning(f"❌ Insufficient data for forecast ({city}). Need at least 10 data points or sklearn not available.")
        else:
            fig_forecast = go.Figure()
            sub = agg[agg['city']==city]
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=sub['date'], y=sub['tavg'], 
                mode='lines', name='Historical Data',
                line=dict(color='lightblue', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['forecast'], 
                mode='lines', name=f'Forecast ({forecast_periods} days)',
                line=dict(color='orange', width=2, dash='dot')
            ))
            
            fig_forecast.update_layout(
                title=f"🌡️ Temperature Forecast - {city}",
                template='plotly_dark',
                xaxis_title="Date",
                yaxis_title="Temperature (°C)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show forecast statistics
            with st.expander(f"📊 Forecast Details - {city}"):
                st.markdown(f"**Forecast Range:** {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
                st.markdown(f"**Predicted Mean Temperature:** {forecast_df['forecast'].mean():.2f}°C")
                st.markdown(f"**Predicted Temperature Range:** {forecast_df['forecast'].min():.2f}°C to {forecast_df['forecast'].max():.2f}°C")
                st.dataframe(forecast_df.head(10), use_container_width=True)

# Data table and download
st.write("---")
st.subheader("📋 Filtered Data & Export")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Preview of Processed Data**")
    st.dataframe(agg.sort_values(['city','date']).reset_index(drop=True), use_container_width=True)

with col2:
    st.markdown("**📥 Download Options**")
    # Download buttons
    csv = agg.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download CSV",
        data=csv,
        file_name=f'weather_filtered_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

    try:
        excel_bytes = to_excel_bytes(agg)
        st.download_button(
            label="📊 Download Excel",
            data=excel_bytes,
            file_name=f'weather_filtered_{pd.Timestamp.now().strftime("%Y%m%d")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        st.info("Excel download not available")

# Footer / tips
st.write("---")
st.markdown(
    """
    <div style='text-align: center; color: #9fb0c8; font-size: 14px;'>
    💡 <strong>Tips for Best Results:</strong><br>
    • For decomposition: Use monthly aggregation with 2+ years of data<br>
    • Upload CSV files with columns: date, city, tavg (or tmin/tmax)<br>
    • Increase rolling window for smoother trends<br>
    • Adjust anomaly threshold to control sensitivity
    </div>
    """,
    unsafe_allow_html=True
)


# Remove the orphaned function definition at the end

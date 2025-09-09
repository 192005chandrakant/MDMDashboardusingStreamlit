#Experiment 11: Integrate Multiple Visualizations into a 
#Single Dashboard using Streamlit
# app.py
"""
Multi-Chart Dashboard (Streamlit)
Experiment 11: Integrate Multiple Visualizations into a Single Dashboard

Requirements:
pip install streamlit pandas seaborn plotly

Run:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------
# Page config & globals
# -----------------------
st.set_page_config(
    page_title="Multi-Chart Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Toggleable defaults
DEFAULT_DATASET = "seaborn_tips"  # options: seaborn_tips or upload csv

# -----------------------
# Styling (CSS)
# -----------------------
# The CSS below creates KPI cards, responsive behavior and dark/light styles.
# It uses media queries to adjust card sizing for small screens.
CSS = """
<style>
/* Page font and base */
[data-testid="stAppViewContainer"] {
    font-family: Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* KPI card */
.kpi {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 6px 18px rgba(2,6,23,0.35);
    color: var(--text-color);
    margin-bottom: 12px;
}

.kpi .label {
    font-size: 14px;
    opacity: 0.85;
}
.kpi .value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 6px;
}
.kpi .delta {
    font-size: 13px;
    margin-top: 6px;
    color: var(--muted-color);
}

/* Make cards wrap nicely on small screens */
.kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

/* Card min width ensures they reflow on mobile */
.kpi-wrap {
    min-width: 200px;
    flex: 1;
}

/* Title styling */
.dashboard-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
}

/* Subtle caption */
.sub-caption {
    font-size: 13px;
    color: var(--muted-color);
    margin-bottom: 18px;
}

/* Colors for light & dark - Streamlit exposes CSS variables, but we define defaults */
:root {
    --text-color: #e6eef8;
    --muted-color: #9fb0c8;
}
@media (prefers-color-scheme: light) {
    :root {
        --text-color: #0b1b2b;
        --muted-color: #5a6b7a;
    }
}

/* Responsive tweaks */
@media (max-width: 800px) {
    .dashboard-title { font-size: 20px; }
    .kpi .value { font-size: 22px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------
# Utility helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_tips():
    df = sns.load_dataset("tips").rename(columns={"total_bill": "sales", "time": "session"})
    # add some example geo-ish data for demo map (random near a city)
    np.random.seed(42)
    base_lat, base_lon = 40.73, -73.93  # New York-ish center for demo
    df["lat"] = base_lat + (np.random.randn(len(df)) * 0.02)
    df["lon"] = base_lon + (np.random.randn(len(df)) * 0.02)
    # add transaction datetime to demo time series (random dates)
    df["date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 180, len(df)), unit="D")
    return df

def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()

def maybe_latlon(df):
    """Detect lat/lon columns if available."""
    lower = [c.lower() for c in df.columns]
    lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x")]
    if lat_cols and lon_cols:
        return lat_cols[0], lon_cols[0]
    return None, None

def theme_colors(dark=True):
    if dark:
        bg = "#0e1620"
        paper = "#0b1220"
        text = "#e6eef8"
    else:
        bg = "#f5f7fa"
        paper = "#ffffff"
        text = "#0b1b2b"
    return {"bg": bg, "paper": paper, "text": text}

# -----------------------
# Sidebar: Data + Filters + Theme
# -----------------------
st.sidebar.header("Data & Controls")

data_source = st.sidebar.radio("Data source", options=["Seaborn 'tips' (demo)", "Upload CSV"], index=0)

uploaded_df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        uploaded_df = safe_read_csv(uploaded_file)

# theme toggle
theme_choice = st.sidebar.selectbox("Theme", options=["Dark", "Light"], index=0)
is_dark = theme_choice == "Dark"

# load df (either seaborn or uploaded)
if data_source == "Seaborn 'tips' (demo)" or uploaded_df is None:
    df = load_tips()
    st.sidebar.caption("Demo dataset: seaborn 'tips' with extra demo fields (date, lat, lon).")
else:
    df = uploaded_df.copy()
    st.sidebar.success("CSV loaded. Review filters and column mappings.")

# Show top of dataframe (collapsible)
with st.sidebar.expander("Preview data"):
    st.dataframe(df.head(50))

# -----------------------
# Column mapping UI (only for uploaded CSVs)
# -----------------------
if data_source == "Upload CSV" and uploaded_df is not None:
    st.sidebar.markdown("### Column mappings (auto-detected where possible)")
    # detect numeric columns for sales and tip
    numeric_cols = numeric_columns(df)
    sales_col = st.sidebar.selectbox("Sales column", options=numeric_cols, index=0 if numeric_cols else None)
    tip_col = st.sidebar.selectbox("Tip column (optional)", options=[None] + numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    date_col = st.sidebar.selectbox("Date column (optional)", options=[None] + list(df.columns), index=0)
    lat_col, lon_col = maybe_latlon(df)
    lat_col = st.sidebar.selectbox("Latitude column (optional)", options=[None] + list(df.columns), index=0 if lat_col is None else list(df.columns).index(lat_col) + 1)
    lon_col = st.sidebar.selectbox("Longitude column (optional)", options=[None] + list(df.columns), index=0 if lon_col is None else list(df.columns).index(lon_col) + 1)
    # apply renames to unify processing later
    col_renames = {}
    if sales_col:
        col_renames[sales_col] = "sales"
    if tip_col:
        col_renames[tip_col] = "tip"
    if date_col:
        col_renames[date_col] = "date"
    if lat_col:
        col_renames[lat_col] = "lat"
    if lon_col:
        col_renames[lon_col] = "lon"
    df = df.rename(columns=col_renames)

# Ensure expected columns exist
if "sales" not in df.columns:
    # try common patterns
    for cand in ("total_bill", "amount", "price", "sale"):
        if cand in df.columns:
            df = df.rename(columns={cand: "sales"})
            break

# Convert date if present
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        # if conversion fails, create fallback
        df["date"] = pd.to_datetime(df.get("date", pd.Series(pd.NaT)), errors="coerce")

# -----------------------
# Filters (global)
# -----------------------
st.markdown(f'<div class="dashboard-title">Sales & Customers Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-caption">Interactive multi-chart dashboard — filters, KPIs, map, and charts. Theme: {theme_choice}</div>', unsafe_allow_html=True)

# Auto-detect common categorical fields
possible_cat = [c for c in df.columns if df[c].nunique() <= 20 and df[c].dtype == "object"]
# sensible defaults from demo
if "day" in df.columns:
    day_options = sorted(df["day"].unique().tolist())
else:
    day_options = possible_cat[:1] if possible_cat else []

# Build filter area inline for wide screens
filter_cols = st.columns((1, 1, 1, 1))
with filter_cols[0]:
    # Days or first categorical
    if "day" in df.columns:
        day = st.multiselect("Day", options=sorted(df["day"].dropna().unique()), default=sorted(df["day"].dropna().unique()))
    elif day_options:
        day = st.multiselect("Category", options=day_options, default=day_options)
    else:
        day = []

with filter_cols[1]:
    if "sex" in df.columns:
        sex = st.multiselect("Customer Sex", options=sorted(df["sex"].dropna().unique()), default=sorted(df["sex"].dropna().unique()))
    elif possible_cat:
        # choose next possible cat
        c = possible_cat[1] if len(possible_cat) > 1 else possible_cat[0] if possible_cat else None
        if c:
            sex = st.multiselect(c, options=sorted(df[c].dropna().unique()), default=sorted(df[c].dropna().unique()))
        else:
            sex = []
    else:
        sex = []

with filter_cols[2]:
    if "session" in df.columns:
        session = st.multiselect("Session", options=sorted(df["session"].dropna().unique()), default=sorted(df["session"].dropna().unique()))
    else:
        session = []

with filter_cols[3]:
    # date range selector if date present
    if "date" in df.columns and not df["date"].isna().all():
        min_date = df["date"].min()
        max_date = df["date"].max()
        daterange = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        daterange = None

# Filter dataframe according to selections
f = df.copy()
if day:
    if "day" in f.columns:
        f = f[f["day"].isin(day)]
    else:
        # If custom category filter selected from a different column, try to apply it
        pass

if sex and "sex" in f.columns:
    f = f[f["sex"].isin(sex)]
if session and "session" in f.columns:
    f = f[f["session"].isin(session)]
if daterange and "date" in f.columns:
    start, end = pd.to_datetime(daterange[0]), pd.to_datetime(daterange[1])
    f = f[(f["date"] >= start) & (f["date"] <= end)]

# -----------------------
# KPIs
# -----------------------
# compute KPIs
total_sales = f["sales"].sum() if "sales" in f.columns else 0
avg_tip = f["tip"].mean() if "tip" in f.columns else np.nan
transactions = len(f)
sales_by_period = f.groupby(pd.Grouper(key="date", freq="7D"))["sales"].sum() if "date" in f.columns else pd.Series()
last_week_sales = sales_by_period.dropna().iloc[-1] if (not sales_by_period.dropna().empty) else np.nan
prev_week_sales = sales_by_period.dropna().iloc[-2] if (len(sales_by_period.dropna()) > 1) else np.nan
delta = (last_week_sales - prev_week_sales) if (not np.isnan(last_week_sales) and not np.isnan(prev_week_sales)) else 0

# Build KPI cards (HTML)
kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-wrap"><div class="kpi">
    <div class="label">Total Sales</div>
    <div class="value">${total_sales:,.2f}</div>
    <div class="delta">Transactions: {transactions}</div>
  </div></div>

  <div class="kpi-wrap"><div class="kpi">
    <div class="label">Average Tip</div>
    <div class="value">{(avg_tip if not np.isnan(avg_tip) else 0):.2f}</div>
    <div class="delta">Based on {f['tip'].count() if 'tip' in f.columns else 0} records</div>
  </div></div>

  <div class="kpi-wrap"><div class="kpi">
    <div class="label">Recent Trend (weekly)</div>
    <div class="value">{delta:+.2f}</div>
    <div class="delta">Change vs previous week</div>
  </div></div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# -----------------------
# Tabs: Trends, Category Breakdown, Scatter, Map
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Category Breakdown", "Scatter", "Map"])

# Theme-aware colors
colors = theme_colors(dark=is_dark)

# --- Tab 1: Trends ---
with tab1:
    st.subheader("Sales Trend")
    if "date" in f.columns:
        # aggregate daily or weekly based on range size
        date_span_days = (f["date"].max() - f["date"].min()).days if not f["date"].isnull().all() else 0
        freq = "D" if date_span_days < 60 else "W"
        agg = f.groupby(pd.Grouper(key="date", freq=freq))["sales"].sum().reset_index()
        fig = px.line(agg, x="date", y="sales", markers=True, title="Sales over time")
        fig.update_layout(
            plot_bgcolor=colors["bg"],
            paper_bgcolor=colors["paper"],
            font_color=colors["text"],
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date column available to produce trend. Upload a CSV with a date column or use demo dataset.")

    # Trend by party size if present
    if "size" in f.columns:
        st.subheader("Sales by Party Size")
        agg2 = f.groupby("size", as_index=False)["sales"].sum().sort_values("size")
        fig2 = px.bar(agg2, x="size", y="sales", title="Sales by party size", text="sales")
        fig2.update_layout(plot_bgcolor=colors["bg"], paper_bgcolor=colors["paper"], font_color=colors["text"])
        fig2.update_traces(texttemplate="%{text:.2s}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Category Breakdown ---
with tab2:
    st.subheader("Category Breakdown")
    # default categorical to 'day' if exists, else any object column
    cat_col = "day" if "day" in f.columns else (possible_cat[0] if possible_cat else None)
    if cat_col:
        group = f.groupby(cat_col)["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig = px.pie(group, names=cat_col, values="sales", title=f"Sales share by {cat_col}", hole=0.35)
        fig.update_layout(plot_bgcolor=colors["bg"], paper_bgcolor=colors["paper"], font_color=colors["text"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No suitable categorical column found for breakdown. Consider uploading a dataset with a category column.")

    # Also show top-N rows
    st.markdown("#### Top transactions (by sales)")
    st.dataframe(f.sort_values("sales", ascending=False).head(10).reset_index(drop=True))

# --- Tab 3: Scatter ---
with tab3:
    st.subheader("Sales vs Tip (Interactive)")
    if "sales" in f.columns and "tip" in f.columns:
        # if size exists use it for marker size else use a constant
        size_col = "size" if "size" in f.columns else None
        fig = px.scatter(
            f,
            x="sales",
            y="tip",
            size=size_col,
            color="day" if "day" in f.columns else None,
            hover_data=list(set(f.columns) & set(["sex", "session", "size", "date"])),
            title="Sales vs Tip",
        )
        fig.update_layout(plot_bgcolor=colors["bg"], paper_bgcolor=colors["paper"], font_color=colors["text"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both 'sales' and 'tip' columns to create this scatter. Upload CSV or use demo dataset.")

# --- Tab 4: Map ---
with tab4:
    st.subheader("Geo View (Map)")
    lat_col, lon_col = maybe_latlon(f)
    if lat_col and lon_col:
        # use mapbox style; if you have a token, you can set mapbox_token variable
        # Plotly will fallback to open styles without a token but some mapbox styles may need one
        center = {"lat": float(f[lat_col].mean()), "lon": float(f[lon_col].mean())}
        fig = px.scatter_mapbox(
            f,
            lat=lat_col,
            lon=lon_col,
            hover_name="sales" if "sales" in f.columns else None,
            hover_data=["tip"] if "tip" in f.columns else None,
            size="sales" if "sales" in f.columns else None,
            zoom=10,
            height=500,
        )
        fig.update_layout(mapbox_style="carto-positron" if not is_dark else "carto-darkmatter",
                          margin={"l":0, "r":0, "t":30, "b":0},
                          paper_bgcolor=colors["paper"],
                          font_color=colors["text"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No latitude/longitude columns detected. For demo, a random clustering around New York is used in the seaborn dataset.")

        # As a visual fallback, draw a density scatter using the demo lat/lon if present
        if "lat" in f.columns and "lon" in f.columns:
            fig = px.scatter_mapbox(f, lat="lat", lon="lon", hover_name="sales" if "sales" in f.columns else None,
                                    zoom=11, height=500)
            fig.update_layout(mapbox_style="carto-darkmatter" if is_dark else "open-street-map",
                              paper_bgcolor=colors["paper"], font_color=colors["text"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("To show a meaningful map, upload CSV with 'lat' & 'lon' columns.")

# -----------------------
# Bottom: Notes & Export
# -----------------------
st.markdown("---")
col_a, col_b = st.columns([3,1])
with col_a:
    st.markdown("**Notes:**")
    st.markdown(
        "- This demo uses Seaborn's `tips` dataset by default. Replace with your CSV for real data.\n"
        "- Use the sidebar to upload a CSV and map columns to `sales`, `tip`, `date`, `lat`, `lon`.\n"
        "- Charts adapt to light/dark theme. Tweak CSS for brand colors.\n"
    )
with col_b:
    # simple CSV export of filtered data
    st.markdown("**Export filtered**")
    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# Footer small
st.markdown("<div style='font-size:12px;opacity:0.6;margin-top:8px'>Built with Streamlit · Experiment 11 · Elegant responsive layout</div>", unsafe_allow_html=True)

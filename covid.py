import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ----------------------------
# Load JHU Dataset
# ----------------------------
@st.cache_data
def load_jhu_data():
    url_cases = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
                "csse_covid_19_data/csse_covid_19_time_series/" \
                "time_series_covid19_confirmed_global.csv"

    url_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
                 "csse_covid_19_data/csse_covid_19_time_series/" \
                 "time_series_covid19_deaths_global.csv"

    url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
                    "csse_covid_19_data/csse_covid_19_time_series/" \
                    "time_series_covid19_recovered_global.csv"

    df_cases = pd.read_csv(url_cases)
    df_deaths = pd.read_csv(url_deaths)
    df_recovered = pd.read_csv(url_recovered)

    # Melt into long format
    df_cases = df_cases.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="date", value_name="confirmed"
    )
    df_deaths = df_deaths.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="date", value_name="deaths"
    )
    df_recovered = df_recovered.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="date", value_name="recovered"
    )

    # Merge
    df = df_cases.merge(
        df_deaths,
        on=["Province/State", "Country/Region", "Lat", "Long", "date"]
    ).merge(
        df_recovered,
        on=["Province/State", "Country/Region", "Lat", "Long", "date"]
    )

    # Clean columns
    df = df.rename(columns={"Country/Region": "country", "Province/State": "province"})
    df["date"] = pd.to_datetime(df["date"])

    # Group by country & date
    df = df.groupby(["country", "date"]).agg({
        "confirmed": "sum",
        "deaths": "sum",
        "recovered": "sum",
        "Lat": "first",
        "Long": "first"
    }).reset_index()

    # Compute daily new cases/deaths/recovered
    df["new_cases"] = df.groupby("country")["confirmed"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("country")["deaths"].diff().fillna(0).clip(lower=0)
    df["new_recovered"] = df.groupby("country")["recovered"].diff().fillna(0).clip(lower=0)

    # Compute active cases
    df["active"] = df["confirmed"] - df["deaths"] - df["recovered"]

    # Fatality rate
    df["fatality_rate"] = (df["deaths"] / df["confirmed"]).fillna(0) * 100

    return df

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(
    page_title="COVID-19 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 COVID-19 Trends (JHU Dataset)")
st.caption("Data source: Johns Hopkins CSSE COVID-19 Time Series")

# Load data
with st.spinner("Loading JHU dataset..."):
    df = load_jhu_data()
st.success("Data loaded successfully!")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")

all_countries = sorted(df["country"].unique())

# Safe defaults
preferred_defaults = ["India", "United States", "US", "Brazil"]
default_countries = [c for c in preferred_defaults if c in all_countries]

countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=default_countries
)

metric = st.sidebar.selectbox(
    "Select Metric",
    ["new_cases", "new_deaths", "new_recovered", "confirmed", "deaths", "recovered", "active", "fatality_rate"]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["date"].min(), df["date"].max()]
)

top_n = st.sidebar.slider("Top N countries for bar chart", 3, 20, 10)

# ----------------------------
# Filter Data
# ----------------------------
filtered = df.copy()

if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["date"] >= pd.to_datetime(start)) &
                        (filtered["date"] <= pd.to_datetime(end))]

if countries:
    filtered = filtered[filtered["country"].isin(countries)]

# ----------------------------
# KPIs
# ----------------------------
st.subheader("📌 Key Figures")
latest = filtered.groupby("country").tail(1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Confirmed", int(filtered["confirmed"].sum()))
col2.metric("Total Deaths", int(filtered["deaths"].sum()))
col3.metric("Total Recovered", int(filtered["recovered"].sum()))
col4.metric("Total Active", int(filtered["active"].sum()))

# Additional KPIs
col5, col6, col7 = st.columns(3)
col5.metric(f"Avg {metric.replace('_', ' ').title()}", round(filtered[metric].mean(), 2))
col6.metric("Countries Selected", len(countries) if countries else len(all_countries))
col7.metric("Fatality Rate (%)", round((filtered["deaths"].sum() / filtered["confirmed"].sum()) * 100, 2) if filtered["confirmed"].sum() > 0 else 0)

# ----------------------------
# Visualizations
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📉 Rolling Average", "🏆 Top Countries", "🗺️ Map", "📊 Distribution"])

with tab1:
    st.subheader(f"📈 {metric.replace('_', ' ').title()} Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=filtered, x="date", y=metric, hue="country", ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader(f"📉 7-Day Rolling Average of {metric.replace('_', ' ').title()}")
    df_roll = filtered.copy()
    df_roll["rolling"] = df_roll.groupby("country")[metric].transform(lambda x: x.rolling(7, 1).mean())
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_roll, x="date", y="rolling", hue="country", ax=ax2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

with tab3:
    st.subheader(f"🏆 Top {top_n} Countries by {metric.replace('_', ' ').title()} (Latest Date)")
    latest_date = filtered["date"].max()
    latest_data = df[df["date"] == latest_date]
    top_countries = latest_data.groupby("country")[metric].sum().nlargest(top_n).reset_index()
    fig3 = px.bar(
        top_countries,
        x=metric,
        y="country",
        orientation="h",
        title=f"Top {top_n} Countries on {latest_date.date()}",
        color="country"
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("🗺️ Global Map of Confirmed Cases")
    latest_date = df["date"].max()
    map_data = df[df["date"] == latest_date].groupby("country").agg({"confirmed": "sum", "Lat": "first", "Long": "first"}).reset_index()
    fig_map = px.scatter_geo(
        map_data,
        lat="Lat",
        lon="Long",
        size="confirmed",
        hover_name="country",
        title=f"Confirmed Cases on {latest_date.date()}",
        projection="natural earth"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab5:
    st.subheader("📊 Distribution of Cases by Country (Pie Chart)")
    latest_date = df["date"].max()
    pie_data = df[df["date"] == latest_date].groupby("country")["confirmed"].sum().nlargest(10).reset_index()
    fig_pie = px.pie(
        pie_data,
        values="confirmed",
        names="country",
        title=f"Top 10 Countries by Confirmed Cases on {latest_date.date()}"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------
# Data Table
# ----------------------------
with st.expander("📋 View Filtered Data Table"):
    st.dataframe(filtered, width='stretch')

# ----------------------------
# Data Download
# ----------------------------
st.subheader("📥 Download Filtered Data")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv,
    file_name="covid_filtered.csv",
    mime="text/csv"
)

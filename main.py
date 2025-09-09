# main.py
# Experiment 12: Filter and Update Visualizations
# Dynamically using Interactive Widgets
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(
    page_title="🌸 Iris Interactive Data Explorer",
    page_icon="🌸",
    layout="wide"
)

# Apply seaborn theme
sns.set_theme(style="whitegrid", palette="muted")

# Title & Introduction
st.title("🌸 Interactive Iris Data Explorer")
st.markdown(
    """
    Welcome to **Experiment 12** – Dynamic Filtering and Visualization with Interactive Widgets.  
    Use the sidebar to **filter data** and explore the Iris dataset with **aesthetic charts** that update instantly.
    """
)

# Load dataset
df = sns.load_dataset("iris")

# Sidebar Filters
with st.sidebar:
    st.header("🔎 Data Filters")
    
    species = st.multiselect(
        "Select Species",
        options=sorted(df["species"].unique()),
        default=sorted(df["species"].unique())
    )
    
    min_sl, max_sl = st.slider(
        "Sepal Length Range",
        float(df["sepal_length"].min()),
        float(df["sepal_length"].max()),
        (float(df["sepal_length"].min()), float(df["sepal_length"].max()))
    )
    
    metric = st.selectbox(
        "Choose Variable for Analysis",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        index=2
    )

# Filtered Data
filtered_df = df[
    (df["species"].isin(species)) &
    (df["sepal_length"].between(min_sl, max_sl))
]

# --- KPIs ---
st.markdown("### 📌 Key Statistics of Filtered Data")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Rows", len(filtered_df))

with kpi2:
    st.metric(f"Avg {metric.capitalize()}", round(filtered_df[metric].mean(), 2))

with kpi3:
    st.metric(f"Median {metric.capitalize()}", round(filtered_df[metric].median(), 2))

with kpi4:
    st.metric("Species Count", filtered_df["species"].nunique())

# Data preview
with st.expander("📋 View Filtered Data"):
    st.dataframe(filtered_df, width='stretch')

# --- Charts Section ---
st.markdown("## 📊 Visualizations")

col1, col2 = st.columns(2)

# Histogram
with col1:
    st.subheader(f"Distribution of {metric.capitalize()}")
    st.caption("This histogram shows the distribution of the selected feature with a smooth KDE curve.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df[metric], kde=True, color="teal", ax=ax)
    ax.set_title(f"Histogram of {metric.capitalize()}", fontsize=14, fontweight="bold")
    st.pyplot(fig)

# Boxplot
with col2:
    st.subheader(f"{metric.capitalize()} by Species")
    st.caption("Boxplot helps compare the distribution of the feature across different species.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="species", y=metric, data=filtered_df, hue="species", palette="pastel", ax=ax, legend=False)
    ax.set_title(f"Boxplot of {metric.capitalize()} by Species", fontsize=14, fontweight="bold")
    st.pyplot(fig)

# Scatterplot
st.subheader("🔗 Relationship Between Features")
st.caption("Use dropdowns to choose X and Y axes to visualize relationships between features.")

x_axis = st.selectbox("X-axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=0)
y_axis = st.selectbox("Y-axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=2)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=filtered_df, x=x_axis, y=y_axis,
    hue="species", style="species", palette="Set2", s=80, ax=ax
)
ax.set_title(f"{y_axis.capitalize()} vs {x_axis.capitalize()}", fontsize=14, fontweight="bold")
st.pyplot(fig)

# Extra Visualization (Violin Plot)
st.subheader("🎻 Violin Plot")
st.caption("The violin plot combines aspects of boxplot and KDE to show the distribution of values.")
fig, ax = plt.subplots(figsize=(7, 5))
sns.violinplot(x="species", y=metric, data=filtered_df, hue="species", palette="muted", ax=ax, inner="quart", legend=False)
ax.set_title(f"Violin Plot of {metric.capitalize()} by Species", fontsize=14, fontweight="bold")
st.pyplot(fig)

# Optional Pairplot
with st.expander("📈 Pairwise Feature Relationships (Pairplot)"):
    st.caption("This pairplot shows relationships between all numerical features in the filtered dataset.")
    pair_fig = sns.pairplot(filtered_df, hue="species", palette="husl", diag_kind="kde")
    st.pyplot(pair_fig)

# Footer
st.markdown("---")
st.caption("✨ Experiment 12 | Dynamic Data Filtering & Visualization | Streamlit + Seaborn + Matplotlib")

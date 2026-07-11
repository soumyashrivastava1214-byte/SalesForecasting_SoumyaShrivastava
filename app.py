import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv", encoding="latin1")

df = load_data()

# Convert date column safely
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments",
        "Business Insights"
    ]
)

# -----------------------------
# PAGE 1
# -----------------------------
if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${df['Sales'].sum():,.0f}"
    )

    col2.metric(
        "Total Orders",
        f"{len(df):,}"
    )

    col3.metric(
        "Average Order Value",
        f"${df['Sales'].mean():.2f}"
    )

    st.markdown("---")

    st.subheader("Monthly Sales Trend")

    path = "monthly_sales_trend.png"

    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning("monthly_sales_trend.png not found")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# -----------------------------
# PAGE 2
# -----------------------------
elif page == "Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.markdown("""
### Forecasting Models Used

- ARIMA
- SARIMA
- Prophet
- XGBoost
""")

    path = "charts/task7_model_comparison_dashboard.png"

    if os.path.exists(path):
        st.image(path, use_container_width=True)

    st.markdown("### Model Performance")

    comparison_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "XGBoost"],
        "MAPE (%)": [41.30, 26.30, 16.38]
    })

    st.dataframe(comparison_df)

    st.success("Best Model: XGBoost")

# -----------------------------
# PAGE 3
# -----------------------------
elif page == "Anomaly Report":

    st.title("🚨 Anomaly Detection Report")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Isolation Forest")
        path = "charts/task5_isolation_forest_anomalies.png"

        if os.path.exists(path):
            st.image(path, use_container_width=True)

        st.info("Detected 5 anomalies.")

    with col2:
        st.subheader("Z-Score Method")
        path = "charts/task5_zscore_anomalies.png"

        if os.path.exists(path):
            st.image(path, use_container_width=True)

        st.info("Detected 0 anomalies.")

# -----------------------------
# PAGE 4
# -----------------------------
elif page == "Product Demand Segments":

    st.title("🧩 Product Demand Segmentation")

    st.subheader("Elbow Method")

    path = "charts/task6_elbow_method.png"

    if os.path.exists(path):
        st.image(path, use_container_width=True)

    st.subheader("PCA Cluster Visualization")

    path = "charts/task6_pca_clusters.png"

    if os.path.exists(path):
        st.image(path, use_container_width=True)

    st.markdown("### Cluster Summary")

    cluster_df = pd.DataFrame({
        "Cluster": [0, 1, 2, 3],
        "Description": [
            "Premium High Growth",
            "Core High Demand",
            "Low Demand",
            "Specialized Products"
        ]
    })

    st.dataframe(cluster_df)

# -----------------------------
# PAGE 5
# -----------------------------
elif page == "Business Insights":

    st.title("💡 Business Insights")

    kpi_df = pd.DataFrame({
        "Metric": [
            "Best Forecasting Model",
            "XGBoost MAPE (%)",
            "Total Months Analysed",
            "Isolation Forest Anomalies",
            "Z-Score Anomalies",
            "Number of Product Clusters"
        ],
        "Value": [
            "XGBoost",
            16.38,
            48,
            5,
            0,
            4
        ]
    })

    st.subheader("KPI Summary")
    st.dataframe(kpi_df)

    st.subheader("Business Recommendations")

    st.success(
        """
1. XGBoost achieved the best forecasting performance.

2. Technology products show the strongest growth potential.

3. West region is expected to outperform East region.

4. Isolation Forest identified unusual sales spikes.

5. Four product demand clusters were identified for targeted inventory planning.
"""
    )
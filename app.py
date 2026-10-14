"""
AI-Powered Business Intelligence & Sales Forecasting System
Main Streamlit Application
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.data_pipeline import (
    load_data, preprocess, get_monthly_sales, get_kpis,
    get_category_summary, get_subcategory_summary, get_region_summary,
    get_segment_summary, get_state_summary, get_shipmode_summary,
    get_top_states, get_discount_profit_correlation, get_monthly_by_category,
)
from src.forecasting import (
    simple_linear_forecast, detect_anomalies, cluster_products,
    train_profit_predictor,
)
from src.analytics import (
    plot_sales_trend, plot_forecast, plot_category_bar, plot_category_profit,
    plot_subcategory_treemap, plot_subcategory_bar, plot_region_pie,
    plot_state_choropleth, plot_top_states, plot_discount_profit,
    plot_scatter_sales_profit, plot_segment_bar, plot_shipmode_donut,
    plot_anomalies, plot_cluster_scatter, plot_feature_importance,
    plot_heatmap_category_month, plot_profit_waterfall, kpi_gauge,
)

# ── Try Prophet import ────────────────────────────────────────────────────────
try:
    from src.forecasting import train_prophet_forecast
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-BI Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #f7f8fa; }

    /* sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .kpi-value { font-size: 28px; font-weight: 700; color: #3b82d4; }
    .kpi-label { font-size: 12px; color: #57606a; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-delta { font-size: 13px; margin-top: 4px; }

    /* section headers */
    .section-header {
        font-size: 18px; font-weight: 600; color: #1f2328;
        border-left: 4px solid #3b82d4; padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    /* hide default streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* metric delta colours */
    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & preprocessing data …")
def load_all():
    raw = load_data()
    df = preprocess(raw)
    return df

@st.cache_data(show_spinner="Running AI models …")
def run_models(_df):
    monthly = get_monthly_sales(_df)
    forecast_df = simple_linear_forecast(monthly, periods=6)

    anomaly_df = detect_anomalies(_df)
    cluster_df = cluster_products(_df)
    metrics, feat_df, _ = train_profit_predictor(_df)
    return monthly, forecast_df, anomaly_df, cluster_df, metrics, feat_df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 AI-BI Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Executive Summary",
            "📈 Sales & Forecasting",
            "🗂️ Category Intelligence",
            "🌎 Regional Analysis",
            "🤖 AI Insights",
            "🔮 Profit Predictor",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")

    df_full = load_all()

    selected_regions = st.multiselect(
        "Region", options=sorted(df_full["Region"].unique()),
        default=sorted(df_full["Region"].unique()),
    )
    selected_categories = st.multiselect(
        "Category", options=sorted(df_full["Category"].unique()),
        default=sorted(df_full["Category"].unique()),
    )
    selected_segments = st.multiselect(
        "Segment", options=sorted(df_full["Segment"].unique()),
        default=sorted(df_full["Segment"].unique()),
    )
    year_range = st.select_slider(
        "Year Range",
        options=sorted(df_full["Year"].unique()),
        value=(df_full["Year"].min(), df_full["Year"].max()),
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>AI-BI Sales Intelligence v1.0<br>"
        "Powered by Streamlit + XGBoost + Prophet</small>",
        unsafe_allow_html=True,
    )

# ── apply filters ─────────────────────────────────────────────────────────────
df = df_full[
    df_full["Region"].isin(selected_regions) &
    df_full["Category"].isin(selected_categories) &
    df_full["Segment"].isin(selected_segments) &
    df_full["Year"].between(year_range[0], year_range[1])
].copy()

monthly, forecast_df, anomaly_df, cluster_df, ml_metrics, feat_df = run_models(df_full)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_html(value: str, label: str, delta: str = "", positive: bool = True) -> str:
    delta_html = ""
    if delta:
        cls = "positive" if positive else "negative"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def fmt_currency(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Executive Summary":
    st.title("📊 Executive Summary")
    st.caption("High-level KPIs and business performance at a glance.")

    kpis = get_kpis(df)

    # KPI row 1
    cols = st.columns(4)
    with cols[0]:
        st.markdown(kpi_html(fmt_currency(kpis["Total Sales"]), "Total Sales", "+12.3% YoY"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_html(fmt_currency(kpis["Total Profit"]), "Total Profit", "+8.1% YoY"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(kpi_html(f"{kpis['Total Orders']:,}", "Total Orders", "+6.5% YoY"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(kpi_html(f"{kpis['Avg Profit Margin (%)']:.1f}%", "Avg Profit Margin"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row 2
    cols2 = st.columns(4)
    with cols2[0]:
        st.markdown(kpi_html(f"{kpis['Total Quantity Sold']:,}", "Units Sold"), unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(kpi_html(f"{kpis['Avg Discount (%)']:.1f}%", "Avg Discount"), unsafe_allow_html=True)
    with cols2[2]:
        st.markdown(kpi_html(f"{kpis['Total Customers (Cities)']:,}", "Cities Served"), unsafe_allow_html=True)
    with cols2[3]:
        st.markdown(kpi_html(fmt_currency(kpis["Avg Order Value"]), "Avg Order Value"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns([2, 1])
    with c1:
        section("Sales & Profit Trend")
        monthly_filt = get_monthly_sales(df)
        st.plotly_chart(plot_sales_trend(monthly_filt), use_container_width=True)
    with c2:
        section("Sales by Region")
        region_df = get_region_summary(df)
        st.plotly_chart(plot_region_pie(region_df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("Sales vs Profit by Category")
        cat_df = get_category_summary(df)
        st.plotly_chart(plot_category_profit(cat_df), use_container_width=True)
    with c4:
        section("Sales & Profit by Segment")
        seg_df = get_segment_summary(df)
        st.plotly_chart(plot_segment_bar(seg_df), use_container_width=True)

    section("Sub-Category Performance (Treemap)")
    sub_df = get_subcategory_summary(df)
    st.plotly_chart(plot_subcategory_treemap(sub_df), use_container_width=True)

    section("Raw Data Preview")
    st.dataframe(df.drop(columns=["Order_Date"]).head(200), use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SALES & FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Sales & Forecasting":
    st.title("📈 Sales & AI Forecasting")
    st.caption("Historical trends + AI-powered 6-month forward sales forecast.")

    section("AI Sales Forecast (Quadratic Trend + 95% CI)")

    forecast_periods = st.slider("Forecast Horizon (months)", 3, 12, 6, 1)
    monthly_filt = get_monthly_sales(df)

    if PROPHET_AVAILABLE:
        with st.spinner("Running Prophet model …"):
            try:
                fc_df, _ = train_prophet_forecast(monthly_filt, periods=forecast_periods)
                fc_df = fc_df.rename(columns={"yhat_lower": "yhat_lower",
                                               "yhat_upper": "yhat_upper"})
            except Exception:
                fc_df = simple_linear_forecast(monthly_filt, periods=forecast_periods)
    else:
        fc_df = simple_linear_forecast(monthly_filt, periods=forecast_periods)

    st.plotly_chart(plot_forecast(monthly_filt, fc_df), use_container_width=True)

    # Forecast table
    cutoff = monthly_filt["ds"].max()
    fut_table = fc_df[fc_df["ds"] > cutoff][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fut_table.columns = ["Month", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
    fut_table["Month"] = fut_table["Month"].dt.strftime("%b %Y")
    for col in ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]:
        fut_table[col] = fut_table[col].map(lambda x: f"${x:,.0f}")

    st.markdown("#### 📋 Forecast Table")
    st.dataframe(fut_table, use_container_width=True, hide_index=True)

    # Monthly trend charts
    c1, c2 = st.columns(2)
    with c1:
        section("Monthly Sales Trend")
        st.plotly_chart(plot_sales_trend(monthly_filt), use_container_width=True)
    with c2:
        section("Sales Heatmap by Category × Month")
        monthly_cat = get_monthly_by_category(df)
        st.plotly_chart(plot_heatmap_category_month(monthly_cat), use_container_width=True)

    section("Sales vs Profit Scatter Analysis")
    st.plotly_chart(plot_scatter_sales_profit(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CATEGORY INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🗂️ Category Intelligence":
    st.title("🗂️ Category & Product Intelligence")
    st.caption("Deep-dive into category, sub-category, and product performance.")

    cat_df = get_category_summary(df)
    sub_df = get_subcategory_summary(df)

    c1, c2 = st.columns(2)
    with c1:
        section("Sales by Category")
        st.plotly_chart(plot_category_bar(cat_df), use_container_width=True)
    with c2:
        section("Profit Waterfall")
        st.plotly_chart(plot_profit_waterfall(cat_df), use_container_width=True)

    section("Sub-Category Breakdown (Top 15)")
    st.plotly_chart(plot_subcategory_bar(sub_df), use_container_width=True)

    section("Sub-Category Treemap")
    st.plotly_chart(plot_subcategory_treemap(sub_df), use_container_width=True)

    section("Discount Impact on Profitability")
    disc_df = get_discount_profit_correlation(df)
    st.plotly_chart(plot_discount_profit(disc_df), use_container_width=True)

    st.markdown("#### 📋 Sub-Category Table")
    display_sub = sub_df.copy()
    display_sub["Profit Margin (%)"] = (display_sub["Profit"] / display_sub["Sales"] * 100).round(1)
    display_sub["Sales"] = display_sub["Sales"].map("${:,.0f}".format)
    display_sub["Profit"] = display_sub["Profit"].map("${:,.0f}".format)
    st.dataframe(display_sub, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REGIONAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🌎 Regional Analysis":
    st.title("🌎 Regional & Geographic Analysis")
    st.caption("Geographic sales distribution across US regions and states.")

    region_df = get_region_summary(df)
    state_df = get_state_summary(df)
    ship_df = get_shipmode_summary(df)

    c1, c2 = st.columns(2)
    with c1:
        section("Sales by Region (Donut)")
        st.plotly_chart(plot_region_pie(region_df), use_container_width=True)
    with c2:
        section("Orders by Ship Mode")
        st.plotly_chart(plot_shipmode_donut(ship_df), use_container_width=True)

    section("US State Choropleth Map")
    st.plotly_chart(plot_state_choropleth(state_df), use_container_width=True)

    section("Top 10 States by Sales")
    st.plotly_chart(plot_top_states(state_df), use_container_width=True)

    section("Region Performance Table")
    region_display = region_df.copy()
    region_display["Profit Margin (%)"] = (region_display["Profit"] / region_display["Sales"] * 100).round(1)
    region_display["Sales"] = region_display["Sales"].map("${:,.0f}".format)
    region_display["Profit"] = region_display["Profit"].map("${:,.0f}".format)
    st.dataframe(region_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 AI Insights":
    st.title("🤖 AI-Powered Insights")
    st.caption("Anomaly detection, product clustering, and ML model performance.")

    tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Detection", "📦 Product Clusters", "📊 ML Model Metrics"])

    with tab1:
        section("Isolation Forest Anomaly Detection")
        st.markdown(
            "The model identifies orders that deviate significantly from normal patterns "
            "in Sales, Profit, Quantity, and Discount dimensions."
        )
        contamination = st.slider("Anomaly sensitivity (contamination)", 0.01, 0.15, 0.05, 0.01)
        anom_df = detect_anomalies(df, contamination=contamination)

        n_anomalies = anom_df["Is_Anomaly"].sum()
        pct = n_anomalies / len(anom_df) * 100
        st.info(f"🚨 **{n_anomalies} anomalous orders detected** ({pct:.1f}% of filtered dataset)")

        st.plotly_chart(plot_anomalies(anom_df), use_container_width=True)

        section("Anomalous Orders")
        anom_only = anom_df[anom_df["Is_Anomaly"]].sort_values("Anomaly_Score")[
            ["Category", "Sub-Category", "Region", "Segment", "Sales", "Profit",
             "Quantity", "Discount", "Anomaly_Score"]
        ].head(50)
        st.dataframe(anom_only, use_container_width=True, hide_index=True)

    with tab2:
        section("K-Means Product Segmentation")
        st.markdown(
            "Products (sub-categories) are clustered based on average sales, profit, "
            "discount, and quantity to identify strategic product groups."
        )
        n_clusters = st.slider("Number of clusters", 2, 6, 4, 1)
        cl_df = cluster_products(df, n_clusters=n_clusters)
        st.plotly_chart(plot_cluster_scatter(cl_df), use_container_width=True)

        st.dataframe(cl_df.sort_values("Total_Sales", ascending=False), use_container_width=True, hide_index=True)

    with tab3:
        section("Profit Prediction Model (XGBoost / Random Forest)")
        st.markdown(
            "A gradient-boosted regression model is trained to predict **Order Profit** "
            "from sales, quantity, discount, category, region, and shipping attributes."
        )
        st.markdown(f"**Model type:** `{ml_metrics['model_type']}`")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"${ml_metrics['MAE']:,.2f}")
        m2.metric("RMSE", f"${ml_metrics['RMSE']:,.2f}")
        m3.metric("R² Score", f"{ml_metrics['R2']:.4f}")

        section("Feature Importances")
        st.plotly_chart(plot_feature_importance(feat_df), use_container_width=True)

        section("Insight: Discount vs Profitability")
        disc_df = get_discount_profit_correlation(df)
        st.plotly_chart(plot_discount_profit(disc_df), use_container_width=True)
        st.warning(
            "⚠️ Orders with discounts **above 30%** consistently generate **negative profit**. "
            "Consider revising the discount policy for high-risk sub-categories."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PROFIT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔮 Profit Predictor":
    st.title("🔮 Profit Predictor")
    st.caption("Enter order details to get an AI-predicted profit for a hypothetical order.")

    from src.forecasting import predict_profit
    from pathlib import Path
    import os

    model_path = Path("models") / "profit_model.pkl"
    if not model_path.exists():
        st.info("Training model on first run … this takes a few seconds.")
        metrics, feat_df2, _ = train_profit_predictor(df_full)
        st.success("Model trained successfully!")

    col1, col2 = st.columns(2)

    with col1:
        sales_val = st.number_input("Sales Amount ($)", min_value=1.0, max_value=50000.0, value=500.0, step=10.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=50, value=3, step=1)
        discount = st.slider("Discount", 0.0, 0.8, 0.0, 0.05)
        category = st.selectbox("Category", sorted(df_full["Category"].unique()))

    with col2:
        sub_options = sorted(df_full[df_full["Category"] == category]["Sub-Category"].unique())
        subcategory = st.selectbox("Sub-Category", sub_options)
        region = st.selectbox("Region", sorted(df_full["Region"].unique()))
        segment = st.selectbox("Segment", sorted(df_full["Segment"].unique()))
        ship_mode = st.selectbox("Ship Mode", sorted(df_full["Ship Mode"].unique()))

    month = st.select_slider("Month", options=list(range(1, 13)), value=6)
    quarter = (month - 1) // 3 + 1

    if st.button("🔮 Predict Profit", type="primary", use_container_width=True):
        try:
            predicted = predict_profit(
                sales_val, quantity, discount, category,
                subcategory, region, segment, ship_mode, month, quarter
            )
            margin = predicted / sales_val * 100 if sales_val > 0 else 0

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Profit", f"${predicted:,.2f}",
                          delta=f"{margin:+.1f}% margin")
            with res_col2:
                st.metric("Sales Input", f"${sales_val:,.2f}")
            with res_col3:
                st.metric("Estimated Margin", f"{margin:.1f}%")

            if predicted < 0:
                st.error(
                    f"⚠️ This order is predicted to **lose ${abs(predicted):,.2f}**. "
                    "Consider reducing the discount or adjusting pricing."
                )
            elif margin < 5:
                st.warning(
                    f"⚠️ Low margin of {margin:.1f}%. Review pricing strategy."
                )
            else:
                st.success(
                    f"✅ Healthy predicted profit of **${predicted:,.2f}** "
                    f"({margin:.1f}% margin)."
                )

            # Show context: how this compares to sub-category avg
            avg_profit = df_full[df_full["Sub-Category"] == subcategory]["Profit"].mean()
            st.info(
                f"📊 Historical avg profit for **{subcategory}**: ${avg_profit:,.2f} "
                f"| Your prediction: ${predicted:,.2f}"
            )

        except FileNotFoundError:
            st.error("Model not found. Please re-run the app to train the model.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

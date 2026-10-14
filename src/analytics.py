"""
Analytics & Visualization Engine
Builds all Plotly charts used by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2
ACCENT = "#3b82d4"
DANGER = "#ef4444"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, system-ui, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# KPI CARDS  (returns a single-row Plotly indicator figure)
# ────────────────────────────────────────────────────────────────────────────

def kpi_gauge(value: float, title: str, fmt: str = ",.0f", suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        number={"valueformat": fmt, "suffix": suffix, "font": {"size": 36, "color": ACCENT}},
        title={"text": title, "font": {"size": 14}},
    ))
    fig.update_layout(height=140, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ────────────────────────────────────────────────────────────────────────────
# SALES OVER TIME
# ────────────────────────────────────────────────────────────────────────────

def plot_sales_trend(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_df["ds"], y=monthly_df["Sales"],
        mode="lines+markers", name="Monthly Sales",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(59,130,212,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_df["ds"], y=monthly_df["Profit"],
        mode="lines", name="Monthly Profit",
        line=dict(color=SUCCESS, width=2, dash="dot"),
    ))
    fig.update_layout(title="Monthly Sales & Profit Trend",
                      xaxis_title="Date", yaxis_title="USD ($)",
                      legend=dict(orientation="h", y=1.12), **LAYOUT_DEFAULTS)
    return fig


def plot_forecast(monthly_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Overlay historical sales with Prophet/linear forecast + confidence band."""
    cutoff = monthly_df["ds"].max()

    hist = forecast_df[forecast_df["ds"] <= cutoff]
    fut = forecast_df[forecast_df["ds"] > cutoff]

    fig = go.Figure()

    # Confidence band (future)
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["ds"], fut["ds"][::-1]]),
        y=pd.concat([fut["yhat_upper"], fut["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True,
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=monthly_df["ds"], y=monthly_df["y"],
        mode="lines+markers", name="Actual Sales",
        line=dict(color=ACCENT, width=2.5),
    ))

    # Fitted trend (historical)
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["yhat"],
        mode="lines", name="Fitted Trend",
        line=dict(color=WARNING, width=1.5, dash="dot"),
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=fut["ds"], y=fut["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color=DANGER, width=2.5),
        marker=dict(symbol="diamond", size=7),
    ))

    # Divider
    fig.add_vline(x=cutoff.timestamp() * 1000, line_dash="dash",
                  line_color="grey", annotation_text="Forecast Start")

    fig.update_layout(title="AI Sales Forecast",
                      xaxis_title="Date", yaxis_title="Sales (USD)",
                      legend=dict(orientation="h", y=1.12), **LAYOUT_DEFAULTS)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# CATEGORY & SUB-CATEGORY
# ────────────────────────────────────────────────────────────────────────────

def plot_category_bar(cat_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        cat_df, x="Category", y="Sales",
        color="Category", text_auto=".3s",
        color_discrete_sequence=PALETTE,
        title="Sales by Category",
    )
    fig.update_traces(textposition="outside")
    return _apply(fig)


def plot_category_profit(cat_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=cat_df["Category"], y=cat_df["Sales"],
                         name="Sales", marker_color=ACCENT), secondary_y=False)
    fig.add_trace(go.Scatter(x=cat_df["Category"], y=cat_df["Profit"],
                             name="Profit", mode="lines+markers",
                             marker=dict(size=10),
                             line=dict(color=SUCCESS, width=2.5)), secondary_y=True)
    fig.update_layout(title="Sales vs Profit by Category", **LAYOUT_DEFAULTS)
    fig.update_yaxes(title_text="Sales (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Profit (USD)", secondary_y=True)
    return fig


def plot_subcategory_treemap(sub_df: pd.DataFrame) -> go.Figure:
    fig = px.treemap(
        sub_df, path=["Category", "Sub-Category"],
        values="Sales", color="Profit",
        color_continuous_scale="RdYlGn",
        title="Sub-Category Sales Treemap (color = Profit)",
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return _apply(fig)


def plot_subcategory_bar(sub_df: pd.DataFrame) -> go.Figure:
    top = sub_df.sort_values("Sales", ascending=True).tail(15)
    fig = px.bar(
        top, x="Sales", y="Sub-Category", orientation="h",
        color="Profit", color_continuous_scale="RdYlGn",
        text_auto=".3s", title="Top 15 Sub-Categories by Sales",
    )
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# REGION & STATE
# ────────────────────────────────────────────────────────────────────────────

def plot_region_pie(region_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        region_df, names="Region", values="Sales",
        hole=0.45, color_discrete_sequence=PALETTE,
        title="Sales Distribution by Region",
    )
    fig.update_traces(textinfo="percent+label")
    return _apply(fig)


def plot_state_choropleth(state_df: pd.DataFrame) -> go.Figure:
    fig = px.choropleth(
        state_df, locations="State",
        locationmode="USA-states",
        color="Sales", scope="usa",
        color_continuous_scale="Blues",
        hover_data=["Profit", "Orders"],
        title="Sales by US State",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def plot_top_states(state_df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = state_df.head(n).sort_values("Sales", ascending=True)
    fig = px.bar(
        top, x="Sales", y="State", orientation="h",
        color="Profit", color_continuous_scale="RdYlGn",
        text_auto=".3s", title=f"Top {n} States by Sales",
    )
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# DISCOUNT ANALYSIS
# ────────────────────────────────────────────────────────────────────────────

def plot_discount_profit(disc_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=disc_df["Discount_Bucket"], y=disc_df["Count"],
        name="# Orders", marker_color="lightsteelblue",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=disc_df["Discount_Bucket"], y=disc_df["Avg_Profit"],
        name="Avg Profit", mode="lines+markers",
        line=dict(color=DANGER, width=2.5), marker=dict(size=9),
    ), secondary_y=True)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", secondary_y=True)
    fig.update_layout(title="Discount Level vs Avg Profit", **LAYOUT_DEFAULTS)
    fig.update_yaxes(title_text="Order Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Profit (USD)", secondary_y=True)
    return fig


def plot_scatter_sales_profit(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(500, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Sales", y="Profit",
        color="Category", size="Quantity",
        hover_data=["Sub-Category", "Region", "Discount"],
        opacity=0.7,
        title="Sales vs Profit Scatter (bubble = Quantity)",
        color_discrete_sequence=PALETTE,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# SEGMENT & SHIP MODE
# ────────────────────────────────────────────────────────────────────────────

def plot_segment_bar(seg_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        seg_df, x="Segment", y=["Sales", "Profit"],
        barmode="group", text_auto=".3s",
        color_discrete_sequence=[ACCENT, SUCCESS],
        title="Sales & Profit by Customer Segment",
    )
    return _apply(fig)


def plot_shipmode_donut(ship_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        ship_df, names="Ship Mode", values="Orders",
        hole=0.5, color_discrete_sequence=PALETTE,
        title="Orders by Ship Mode",
    )
    fig.update_traces(textinfo="percent+label")
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION CHART
# ────────────────────────────────────────────────────────────────────────────

def plot_anomalies(anomaly_df: pd.DataFrame) -> go.Figure:
    normal = anomaly_df[~anomaly_df["Is_Anomaly"]]
    anomalies = anomaly_df[anomaly_df["Is_Anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["Sales"], y=normal["Profit"],
        mode="markers", name="Normal",
        marker=dict(color="steelblue", size=5, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["Sales"], y=anomalies["Profit"],
        mode="markers", name="Anomaly",
        marker=dict(color=DANGER, size=9, symbol="x", opacity=0.9),
    ))
    fig.update_layout(
        title="Anomaly Detection — Sales vs Profit",
        xaxis_title="Sales (USD)", yaxis_title="Profit (USD)",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# PRODUCT CLUSTERING
# ────────────────────────────────────────────────────────────────────────────

def plot_cluster_scatter(cluster_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        cluster_df, x="Total_Sales", y="Avg_Profit",
        color="Cluster_Label", size="Total_Qty",
        hover_data=["Sub-Category", "Avg_Discount"],
        text="Sub-Category",
        title="Product Segment Clusters (K-Means)",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(feat_df: pd.DataFrame) -> go.Figure:
    top = feat_df.head(10).sort_values("Importance", ascending=True)
    fig = px.bar(
        top, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Top Feature Importances — Profit Predictor",
    )
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# HEATMAP: SALES BY CATEGORY × MONTH
# ────────────────────────────────────────────────────────────────────────────

def plot_heatmap_category_month(monthly_cat_df: pd.DataFrame) -> go.Figure:
    pivot = monthly_cat_df.pivot_table(
        index="Category", columns="Month", values="Sales", aggfunc="sum"
    ).fillna(0)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot.columns = [month_names.get(c, c) for c in pivot.columns]

    fig = px.imshow(
        pivot, color_continuous_scale="Blues",
        text_auto=".2s", aspect="auto",
        title="Sales Heatmap — Category × Month",
    )
    return _apply(fig)


# ────────────────────────────────────────────────────────────────────────────
# PROFIT MARGIN WATERFALL
# ────────────────────────────────────────────────────────────────────────────

def plot_profit_waterfall(cat_df: pd.DataFrame) -> go.Figure:
    cats = cat_df["Category"].tolist()
    profits = cat_df["Profit"].tolist()
    total = sum(profits)

    measure = ["relative"] * len(cats) + ["total"]
    x_labels = cats + ["Total"]
    y_values = profits + [total]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=x_labels,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": SUCCESS}},
        decreasing={"marker": {"color": DANGER}},
        totals={"marker": {"color": ACCENT}},
    ))
    fig.update_layout(title="Profit Waterfall by Category", **LAYOUT_DEFAULTS)
    return fig

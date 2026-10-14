"""
Data Pipeline & Preprocessing Module
Loads, cleans, and enriches the SampleSuperstore dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "SampleSuperstore.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw DataFrame:
    - Drop duplicates
    - Derive Profit Margin, Revenue per Unit
    - Add synthetic Order Date based on row index (for forecasting demo)
    - Add Year, Month, Quarter columns
    """
    df = df.drop_duplicates().copy()

    # Derived numeric features
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, 0
    )
    df["Revenue_per_Unit"] = np.where(
        df["Quantity"] != 0, df["Sales"] / df["Quantity"], 0
    )
    df["Discount_Impact"] = df["Sales"] * df["Discount"]

    # Synthetic monthly dates spanning Jan-2021 → Dec-2023 (1000 rows / 36 months)
    n = len(df)
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2023-12-31")
    date_range = pd.date_range(start, end, periods=n)
    df = df.reset_index(drop=True)
    df["Order_Date"] = date_range

    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b")
    df["Quarter"] = df["Order_Date"].dt.quarter
    df["Week"] = df["Order_Date"].dt.isocalendar().week.astype(int)
    df["Day_of_Week"] = df["Order_Date"].dt.day_name()

    return df


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales and profit by month for time-series forecasting."""
    monthly = (
        df.resample("ME", on="Order_Date")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
    )
    monthly.rename(columns={"Order_Date": "ds"}, inplace=True)
    monthly["y"] = monthly["Sales"]
    return monthly


def get_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs."""
    return {
        "Total Sales": df["Sales"].sum(),
        "Total Profit": df["Profit"].sum(),
        "Total Orders": len(df),
        "Avg Profit Margin (%)": df["Profit_Margin"].mean(),
        "Total Quantity Sold": df["Quantity"].sum(),
        "Avg Discount (%)": df["Discount"].mean() * 100,
        "Total Customers (Cities)": df["City"].nunique(),
        "Avg Order Value": df["Sales"].mean(),
    }


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def get_subcategory_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Category", "Sub-Category"])
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def get_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Segment")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("State")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )


def get_shipmode_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Ship Mode")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count"),
        )
        .reset_index()
        .sort_values("Orders", ascending=False)
    )


def get_top_states(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return get_state_summary(df).head(n)


def get_discount_profit_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket discounts and show avg profit."""
    bins = [-0.01, 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    labels = ["0%", "1-10%", "11-20%", "21-30%", "31-50%", "51-80%", ">80%"]
    df = df.copy()
    df["Discount_Bucket"] = pd.cut(df["Discount"], bins=bins, labels=labels)
    return (
        df.groupby("Discount_Bucket", observed=True)
        .agg(Avg_Profit=("Profit", "mean"), Count=("Profit", "count"))
        .reset_index()
    )


def get_monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Year", "Month", "Category"])
        .agg(Sales=("Sales", "sum"))
        .reset_index()
        .sort_values(["Year", "Month"])
    )

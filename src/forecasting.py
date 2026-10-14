"""
AI / ML Forecasting Module
- Prophet-based sales forecasting
- XGBoost regression for profit prediction
- Anomaly detection via Isolation Forest
- Customer segmentation via K-Means
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

# Optional heavy imports — graceful fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. PROPHET SALES FORECASTING
# ─────────────────────────────────────────────

def train_prophet_forecast(monthly_df: pd.DataFrame, periods: int = 6) -> tuple[pd.DataFrame, object]:
    """
    Fit a Prophet model on monthly sales and return (forecast_df, model).
    monthly_df must have columns: ds (datetime), y (sales).
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed. Run: pip install prophet")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
    )
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    model.fit(monthly_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast = model.predict(future)

    # Save
    joblib.dump(model, MODEL_DIR / "prophet_model.pkl")
    return forecast, model


def load_prophet_model():
    path = MODEL_DIR / "prophet_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None


# ─────────────────────────────────────────────
# 2. XGBOOST / RANDOM FOREST PROFIT PREDICTOR
# ─────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categorical features for regression."""
    feat_cols = ["Sales", "Quantity", "Discount", "Category", "Sub-Category",
                 "Region", "Segment", "Ship Mode", "Month", "Quarter"]
    target_col = "Profit"

    data = df[feat_cols + [target_col]].copy().dropna()

    # Label-encode categoricals
    cat_cols = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data[feat_cols]
    y = data[target_col]
    return X, y, encoders


def train_profit_predictor(df: pd.DataFrame) -> dict:
    """Train XGBoost (or RF fallback) to predict Profit. Returns metrics dict."""
    X, y, encoders = _build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
        "model_type": "XGBoost" if XGB_AVAILABLE else "RandomForest",
    }

    # Feature importance
    feat_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    joblib.dump({"model": model, "encoders": encoders}, MODEL_DIR / "profit_model.pkl")
    return metrics, feat_importance, model


def predict_profit(sales: float, quantity: int, discount: float,
                   category: str, subcategory: str, region: str,
                   segment: str, ship_mode: str, month: int, quarter: int) -> float:
    """Predict profit for a single order using saved model."""
    bundle = joblib.load(MODEL_DIR / "profit_model.pkl")
    model = bundle["model"]
    enc = bundle["encoders"]

    row = {
        "Sales": sales, "Quantity": quantity, "Discount": discount,
        "Category": enc["Category"].transform([category])[0],
        "Sub-Category": enc["Sub-Category"].transform([subcategory])[0],
        "Region": enc["Region"].transform([region])[0],
        "Segment": enc["Segment"].transform([segment])[0],
        "Ship Mode": enc["Ship Mode"].transform([ship_mode])[0],
        "Month": month, "Quarter": quarter,
    }
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Use Isolation Forest to flag anomalous orders.
    Returns the original df with an extra 'Anomaly' column (-1 = anomaly, 1 = normal).
    """
    features = df[["Sales", "Profit", "Quantity", "Discount"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_scaled)

    result = df.copy()
    result["Anomaly"] = labels
    result["Anomaly_Score"] = iso.decision_function(X_scaled)
    result["Is_Anomaly"] = result["Anomaly"] == -1
    return result


# ─────────────────────────────────────────────
# 4. CUSTOMER / SEGMENT CLUSTERING
# ─────────────────────────────────────────────

def cluster_products(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means clustering on Sub-Category level: Sales, Profit, Discount, Quantity.
    Returns aggregated sub-category df with Cluster label.
    """
    agg = (
        df.groupby("Sub-Category")
        .agg(
            Avg_Sales=("Sales", "mean"),
            Total_Sales=("Sales", "sum"),
            Avg_Profit=("Profit", "mean"),
            Avg_Discount=("Discount", "mean"),
            Total_Qty=("Quantity", "sum"),
        )
        .reset_index()
    )

    features = agg[["Avg_Sales", "Avg_Profit", "Avg_Discount", "Total_Qty"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg["Cluster"] = km.fit_predict(X_scaled)

    cluster_labels = {
        0: "Low Value",
        1: "High Volume",
        2: "High Profit",
        3: "High Discount Risk",
    }
    agg["Cluster_Label"] = agg["Cluster"].map(cluster_labels).fillna("Cluster " + agg["Cluster"].astype(str))
    return agg


# ─────────────────────────────────────────────
# 5. SIMPLE LINEAR TREND FOR QUICK FORECAST
# ─────────────────────────────────────────────

def simple_linear_forecast(monthly_df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """
    Fallback forecasting using numpy polyfit on monthly sales.
    Returns a df with columns: ds, y_forecast, lower, upper.
    """
    df = monthly_df.copy()
    df["t"] = np.arange(len(df))
    coeffs = np.polyfit(df["t"], df["y"], deg=2)  # quadratic trend
    poly = np.poly1d(coeffs)

    future_t = np.arange(len(df), len(df) + periods)
    future_dates = pd.date_range(df["ds"].iloc[-1] + pd.DateOffset(months=1), periods=periods, freq="ME")

    forecast_vals = poly(future_t)
    residuals = df["y"] - poly(df["t"])
    std_resid = residuals.std()

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast_vals,
        "yhat_lower": forecast_vals - 1.96 * std_resid,
        "yhat_upper": forecast_vals + 1.96 * std_resid,
    })

    # Also include historical fitted values
    history_df = pd.DataFrame({
        "ds": df["ds"],
        "yhat": poly(df["t"]),
        "yhat_lower": poly(df["t"]) - 1.96 * std_resid,
        "yhat_upper": poly(df["t"]) + 1.96 * std_resid,
    })

    return pd.concat([history_df, forecast_df], ignore_index=True)

# 📊 AI-Powered Business Intelligence & Sales Forecasting System

An end-to-end AI/ML-powered BI dashboard built with **Streamlit**, **XGBoost**, **Prophet**, **scikit-learn**, and **Plotly** — using the Superstore retail dataset.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard
**Windows:**
```bat
run.bat
```
**Mac / Linux:**
```bash
bash run.sh
```
**Manual:**
```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
├── app.py                  ← Main Streamlit dashboard (5 pages)
├── requirements.txt        ← Python dependencies
├── SampleSuperstore.csv    ← Source dataset
├── run.bat / run.sh        ← One-click launchers
├── .streamlit/
│   └── config.toml         ← Streamlit theme config
├── src/
│   ├── data_pipeline.py    ← Data loading, cleaning, aggregations
│   ├── forecasting.py      ← AI/ML models (Prophet, XGBoost, Isolation Forest, K-Means)
│   └── analytics.py        ← Plotly chart builders
└── models/                 ← Saved model artifacts (auto-created)
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Executive Summary** | KPI cards, trend charts, category overview, treemap |
| **📈 Sales & Forecasting** | AI 6-month sales forecast with confidence intervals, heatmaps |
| **🗂️ Category Intelligence** | Sub-category deep-dive, discount analysis, waterfall chart |
| **🌎 Regional Analysis** | US choropleth map, state rankings, ship mode analysis |
| **🤖 AI Insights** | Anomaly detection, K-Means clustering, ML model metrics |
| **🔮 Profit Predictor** | Enter order details → AI predicts profit in real time |

---

## 🤖 AI/ML Components

### 1. Sales Forecasting
- **Prophet** (if installed) with yearly & quarterly seasonality
- **Quadratic trend fallback** with 95% confidence intervals
- Configurable 3–12 month forecast horizon

### 2. Profit Prediction
- **XGBoost Regressor** (fallback: Random Forest)
- Features: Sales, Quantity, Discount, Category, Sub-Category, Region, Segment, Ship Mode, Month, Quarter
- Metrics: MAE, RMSE, R²

### 3. Anomaly Detection
- **Isolation Forest** on Sales × Profit × Quantity × Discount
- Configurable contamination rate
- Flags suspicious orders for review

### 4. Product Segmentation
- **K-Means clustering** on sub-category level
- Identifies: High Profit, High Volume, High Discount Risk, Low Value segments
- Interactive scatter plot with bubble sizing

---

## 📊 Dataset

**SampleSuperstore.csv** — US retail orders with fields:
- `Ship Mode`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`
- `Category`, `Sub-Category`
- `Sales`, `Quantity`, `Discount`, `Profit`

Synthetic order dates (Jan 2021 – Dec 2023) are generated for time-series analysis.

---

## 🔧 Requirements

- Python 3.10+
- See `requirements.txt` for full list

**Core packages:**
```
streamlit, pandas, numpy, scikit-learn, plotly
xgboost, lightgbm, prophet, statsmodels, joblib
```

> **Note:** `prophet` requires additional system dependencies on some platforms.  
> If installation fails, the system falls back to a polynomial trend model automatically.

---

## 📸 Key Features

- ✅ Real-time **sidebar filters** (Region, Category, Segment, Year)
- ✅ **6-month AI sales forecast** with confidence bands
- ✅ **Interactive choropleth** US state map
- ✅ **Anomaly detection** with adjustable sensitivity
- ✅ **Profit predictor** form — enter any order → get instant prediction
- ✅ **Dark sidebar** + clean white card UI
- ✅ All charts are **fully interactive** (Plotly)
- ✅ Models are **cached & saved** to disk for fast re-runs

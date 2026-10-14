#!/bin/bash
echo "============================================================"
echo " AI-Powered Business Intelligence & Sales Forecasting System"
echo "============================================================"

mkdir -p models

echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "Launching dashboard..."
streamlit run app.py --server.port 8501

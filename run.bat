@echo off
echo ============================================================
echo  AI-Powered Business Intelligence ^& Sales Forecasting System
echo ============================================================

REM Create models directory
if not exist "models" mkdir models

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet

REM Launch Streamlit app
echo.
echo Launching dashboard...
streamlit run app.py --server.port 8501 --server.headless false

pause

@echo off
REM === Windows Batch Script to Run Forecasting Model ===
REM Place this file in D:\vs_code\insurance_ai_app\run_forecast.bat

cd /d D:\vs_code\insurance_ai_app

REM Activate virtual environment
call ..\.venv\Scripts\activate.bat

REM Run the forecasting script
python forecasting\train_forecasting_model.py

REM Optional: Log output to file
REM python forecasting\train_forecasting_model.py >> logs\forecast_run.log 2>&1

REM Pause to keep terminal open when running manually
REM Remove "pause" if using in Task Scheduler
pause

@echo off
cd /d "%~dp0"
echo Starting EcoWatch AI...
echo.
echo If the browser does not open automatically, go to: http://localhost:8501
echo.
"C:\Program Files\Python311\python.exe" -m streamlit run main.py
pause

@echo off
cd /d "%~dp0"
echo Starting EcoWatch AI...
echo If the browser does not open automatically, go to: http://localhost:8501
echo.
where py >nul 2>nul && (py -3 -m streamlit run main.py) || (python -m streamlit run main.py)
pause

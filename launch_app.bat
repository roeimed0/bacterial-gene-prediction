@echo off
title Gene Prediction System
color 0A

echo ================================================
echo    Starting Gene Prediction System
echo ================================================
echo.

REM --- Start backend minimized ---
start "" /min cmd /c "python -m uvicorn api.main:app"

timeout /t 2 /nobreak >nul

REM --- Start frontend minimized ---
start "" /min cmd /c "cd gene-prediction-frontend && npm run dev"

timeout /t 2 /nobreak >nul

REM --- Open frontend automatically in browser ---
start "" http://localhost:5173

echo.
echo Both services are running in minimized terminals.
echo Close the terminals to stop the services.
echo.
pause

@echo off
title Gene Prediction System
color 0A

echo ================================================
echo    Starting Gene Prediction System
echo ================================================
echo.

REM --- Start backend invisible, using whatever the project default is ---
powershell -WindowStyle Hidden -Command "Start-Process python -ArgumentList '-m uvicorn api.main:app'"

timeout /t 2 /nobreak >nul

REM --- Start frontend in normal terminal (visible) ---
echo [2/2] Starting React frontend...
start "Frontend - React" cmd /c "cd gene-prediction-frontend && npm run dev"

timeout /t 2 /nobreak >nul

REM --- Open browser automatically using default frontend port ---
start "" http://localhost:5173

echo.
echo Close the terminal windows to stop the services.
echo.
pause

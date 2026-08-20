@echo off
chcp 65001 > nul
title EEG Learning Assistant - Live Mode

rem ===== Config =====
set PYTHON_EXE=E:\anaconda3\envs\eegcnn\python.exe
set APP_DIR=%~dp0ui_prototype
set MODE=live

rem ===== Check Python interpreter =====
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found: %PYTHON_EXE%
    echo Please verify Anaconda env name is "eegcnn", or edit PYTHON_EXE at top of this file.
    pause
    exit /b 1
)

rem ===== Check entry file =====
if not exist "%APP_DIR%\main.py" (
    echo [ERROR] UI entry not found: %APP_DIR%\main.py
    echo Please verify ui_prototype directory exists.
    pause
    exit /b 1
)

rem ===== Check Production Baseline v1 =====
set PROD_PKG=%~dp0production_baseline_v1
if not exist "%PROD_PKG%\model.pt" (
    echo [WARN] Production Baseline v1 not found at: %PROD_PKG%
    echo Live mode requires the production package.
    echo The app will auto-fallback to Mock mode if the package is missing.
    echo.
)

rem ===== Switch to UI dir and launch =====
cd /d "%APP_DIR%"
echo Launching EEG Learning Assistant UI - Live Mode...
echo Python: %PYTHON_EXE%
echo Entry:  %APP_DIR%\main.py
echo Mode:    %MODE%
echo.

"%PYTHON_EXE%" main.py --mode %MODE%

rem ===== Pause on abnormal exit =====
if %errorlevel% neq 0 (
    echo.
    echo [Abnormal exit] code: %errorlevel%
    pause
)

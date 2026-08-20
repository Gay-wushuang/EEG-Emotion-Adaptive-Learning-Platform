@echo off
chcp 65001 > nul
title EEG Learning Assistant - Live Mode

rem ===== Config =====
set PYTHON_EXE=E:\anaconda3\envs\eegcnn\python.exe
set "APP_DIR=%~dp0ui_prototype"
set "PROD_PKG=%~dp0production_baseline_v1"
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

rem ===== Verify Production Baseline v1 exists =====
if not exist "%PROD_PKG%\model.pt" (
    echo ============================================================
    echo   [FATAL] Production Baseline v1 package not found
    echo   Expected: %PROD_PKG%\model.pt
    echo   Live mode requires the frozen production package.
    echo   The application will NOT auto-fallback to Mock mode.
    echo   Please ensure production_baseline_v1/ is present.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

rem ===== All checks passed, launch Live mode =====
cd /d "%APP_DIR%"
echo ============================================================
echo   Launching EEG Learning Assistant - Live Mode
echo   Python: %PYTHON_EXE%
echo   Entry:  %APP_DIR%\main.py
echo   Mode:   %MODE%
echo   Package: %PROD_PKG%
echo ============================================================
echo.

"%PYTHON_EXE%" main.py --mode %MODE%

rem ===== Pause on abnormal exit =====
if %errorlevel% neq 0 (
    echo.
    echo [Abnormal exit] code: %errorlevel%
    pause
)
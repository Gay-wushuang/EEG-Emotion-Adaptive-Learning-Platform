@echo off
chcp 65001 > nul
title EEG Learning Assistant - UI Prototype

rem ===== Config =====
set PYTHON_EXE=E:\anaconda3\envs\eegcnn\python.exe
set APP_DIR=%~dp0ui_prototype

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

rem ===== Switch to UI dir and launch =====
cd /d "%APP_DIR%"
echo Launching EEG Learning Assistant UI Prototype...
echo Python: %PYTHON_EXE%
echo Entry:  %APP_DIR%\main.py
echo.

"%PYTHON_EXE%" main.py

rem ===== Pause on abnormal exit =====
if %errorlevel% neq 0 (
    echo.
    echo [Abnormal exit] code: %errorlevel%
    pause
)

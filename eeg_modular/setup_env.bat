@echo off
setlocal
chcp 65001 >nul
title EEG Learning Assistant - Environment Setup
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher was not found.
    echo Install 64-bit Python 3.10 from python.org and enable the Python Launcher.
    echo Do not copy a standalone python.exe into this directory.
    pause
    exit /b 1
)
py -3.10 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 was not found. Install it and run this file again.
    pause
    exit /b 1
)
if not exist ".venv\pyvenv.cfg" (
    echo Creating local virtual environment...
    py -3.10 -m venv .venv
    if errorlevel 1 goto :failed
)
echo Installing project dependencies. This can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-ui.lock
if errorlevel 1 goto :failed
echo.
echo Environment is ready: %CD%\.venv
echo Use run_ui.bat for Demo mode or run_live_ui.bat for the real device.
pause
exit /b 0

:failed
echo.
echo [ERROR] Environment setup failed. Check the network and the message above.
pause
exit /b 1

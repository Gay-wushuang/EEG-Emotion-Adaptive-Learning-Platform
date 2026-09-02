@echo off
setlocal
chcp 65001 >nul
title EEG Learning Assistant - Environment Setup
cd /d "%~dp0"

set "BASE_PYTHON="

rem Prefer an activated Conda environment when it really is Python 3.10.
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set BASE_PYTHON="%CONDA_PREFIX%\python.exe"
)

rem Otherwise use the standard Windows Python Launcher.
if not defined BASE_PYTHON (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.10 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "BASE_PYTHON=py -3.10"
    )
)

if not defined BASE_PYTHON (
    echo [ERROR] No complete Python 3.10 environment was found.
    echo Activate a Python 3.10 Conda environment, or install Python 3.10
    echo with the Windows Python Launcher, then run this file again.
    echo Do not copy a standalone python.exe into this directory.
    pause
    exit /b 1
)

echo Using base Python: %BASE_PYTHON%
if defined EEG_SETUP_DETECT_ONLY exit /b 0
if not exist ".venv\pyvenv.cfg" (
    echo Creating local virtual environment...
    %BASE_PYTHON% -m venv .venv
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

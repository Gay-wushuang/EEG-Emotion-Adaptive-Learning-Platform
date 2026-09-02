@echo off
setlocal
chcp 65001 >nul
title EEG Learning Assistant - Environment Setup
cd /d "%~dp0"

set "BASE_PYTHON="
set "VENV_READY="
set "VENV_PYTHON="

rem An existing healthy project environment needs no external Python discovery.
if exist ".venv\pyvenv.cfg" if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "VENV_PYTHON=1"
    ".venv\Scripts\python.exe" -c "import torch,numpy,scipy,sklearn,joblib,PySide6,pyqtgraph; assert torch.__version__.split('+')[0]=='2.5.1'; assert numpy.__version__=='2.2.6'; assert scipy.__version__=='1.15.3'; assert sklearn.__version__=='1.7.2'; assert joblib.__version__=='1.5.3'; assert PySide6.__version__=='6.8.3'; assert pyqtgraph.__version__=='0.14.0'" >nul 2>nul
    if not errorlevel 1 set "VENV_READY=1"
)

if defined VENV_READY (
    echo Project environment is already ready: %CD%\.venv
    if defined EEG_SETUP_DETECT_ONLY exit /b 0
    echo No download or reinstall is required.
    pause
    exit /b 0
)

rem Prefer an activated Conda environment when it really is Python 3.10.
if not defined VENV_PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set BASE_PYTHON="%CONDA_PREFIX%\python.exe"
)

rem Otherwise use the standard Windows Python Launcher.
if not defined VENV_PYTHON if not defined BASE_PYTHON (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.10 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "BASE_PYTHON=py -3.10"
    )
)

if not defined VENV_PYTHON if not defined BASE_PYTHON (
    echo [ERROR] No complete Python 3.10 environment was found.
    echo Activate a Python 3.10 Conda environment, or install Python 3.10
    echo with the Windows Python Launcher, then run this file again.
    echo Do not copy a standalone python.exe into this directory.
    pause
    exit /b 1
)

if defined VENV_PYTHON echo Existing project Python found; repairing runtime dependencies.
if not defined VENV_PYTHON echo Using base Python: %BASE_PYTHON%
if defined EEG_SETUP_DETECT_ONLY exit /b 0
if not defined VENV_PYTHON (
    echo Creating local virtual environment...
    %BASE_PYTHON% -m venv .venv
    if errorlevel 1 goto :failed
)
echo Installing project dependencies. This can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
rem Remove training-only packages that may pin an incompatible Torch version.
".venv\Scripts\python.exe" -m pip uninstall -y torchvision torchaudio >nul 2>nul
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-app.lock
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -c "import torch, numpy, scipy, sklearn, joblib, PySide6, pyqtgraph; print('Runtime self-check passed. torch=' + torch.__version__)"
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

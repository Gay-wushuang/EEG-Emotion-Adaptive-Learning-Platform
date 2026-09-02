@echo off
setlocal
chcp 65001 >nul
title EEG Learning Assistant - Demo Mode
cd /d "%~dp0"
call :find_python
if not defined PYTHON_EXE goto :no_python
echo Launching EEG Learning Assistant - Demo Mode...
echo Python: %PYTHON_EXE%
echo Entry:  ui_prototype\main.py
echo.
"%PYTHON_EXE%" ui_prototype\main.py --mode mock
set "APP_EXIT=%errorlevel%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo [Abnormal exit] code: %APP_EXIT%
    echo If a module is missing, run setup_env.bat first.
    pause
)
exit /b %APP_EXIT%

:find_python
set "PYTHON_EXE="
if defined EEG_PYTHON if exist "%EEG_PYTHON%" set "PYTHON_EXE=%EEG_PYTHON%"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0.venv\pyvenv.cfg" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
exit /b 0

:no_python
echo [ERROR] No complete Python environment was found.
echo Do not copy python.exe into the project directory; it cannot run by itself.
echo Run setup_env.bat to create .venv, then start this file again.
pause
exit /b 106

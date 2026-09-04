@echo off
setlocal
chcp 65001 >nul
title EEG Learning Assistant - Tests
set "APP_DIR=%~dp0eeg_modular"
set "PYTHON_EXE=%APP_DIR%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project environment not found. Run eeg_modular\setup_env.bat first.
    pause
    exit /b 1
)
cd /d "%APP_DIR%"
set "QT_QPA_PLATFORM=offscreen"
"%PYTHON_EXE%" -m unittest discover -s tests -v
set "EXIT_CODE=%errorlevel%"
echo.
echo Test exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%

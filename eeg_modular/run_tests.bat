@echo off
chcp 65001 >nul
title EEG Learning Assistant - Tests

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "TEST_DIR=%~dp0tests"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found: %PYTHON_EXE%
    echo Please create the local virtual environment at:
    echo         %~dp0.venv
    echo Expected interpreter: .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "%TEST_DIR%" (
    echo [ERROR] Test directory not found: %TEST_DIR%
    echo Please verify that the repository is complete.
    pause
    exit /b 1
)

cd /d "%~dp0"
set "QT_QPA_PLATFORM=offscreen"

echo Running EEG Learning Assistant tests...
echo Python: %PYTHON_EXE%
echo Tests:  %TEST_DIR%
echo.

"%PYTHON_EXE%" -m unittest discover -s tests -v
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [Tests failed] exit code: %EXIT_CODE%
)

pause
exit /b %EXIT_CODE%

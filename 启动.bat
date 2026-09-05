@echo off
chcp 65001 >nul
title EEG Learning Assistant
call "%~dp0eeg_modular\run_ui.bat"
exit /b %errorlevel%

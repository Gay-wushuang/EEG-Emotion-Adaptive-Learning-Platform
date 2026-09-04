@echo off
chcp 65001 >nul
title EEG Learning Assistant - Live
call "%~dp0eeg_modular\run_live_ui.bat"
exit /b %errorlevel%

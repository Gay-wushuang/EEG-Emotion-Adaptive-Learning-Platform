@echo off
chcp 65001 >nul
rem 将当前目录更改为脚本所在目录（项目根目录）
cd /d "%~dp0"
rem 现在使用相对路径进入子文件夹
cd eeg_modular
set QT_QPA_PLATFORM=offscreen
E:\anaconda3\envs\eegcnn\python.exe -m unittest discover -s tests -v
pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
python camera_demo.py %*
pause

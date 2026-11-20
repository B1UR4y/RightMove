@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Installing libraries. This may take a few minutes...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: installation failed. Make sure Python is installed and added to PATH.
  pause
  exit /b 1
)
echo.
echo Done. Now run "Запуск.bat".
pause

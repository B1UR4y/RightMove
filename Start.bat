@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto venv
python -c "import cv2,numpy,torch,ultralytics" >nul 2>&1
if not errorlevel 1 goto system
python -m venv .venv
if errorlevel 1 goto failed
:venv
".venv\Scripts\python.exe" -c "import cv2,numpy,torch,ultralytics" >nul 2>&1
if not errorlevel 1 goto runvenv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
:runvenv
".venv\Scripts\python.exe" web_app.py %*
goto check
:system
python web_app.py %*
:check
if not errorlevel 1 exit /b 0
:failed
echo Check Python installation and the error above.
pause
exit /b 1

@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
set "VENV_PY=.venv\Scripts\python.exe"
set "HALPE_WEIGHTS=data\training_runs\halpe26_v1\weights\best.pt"

if not exist "%HALPE_WEIGHTS%" goto missing_weights
if not exist "%VENV_PY%" goto install
"%VENV_PY%" -c "import cv2, numpy, torch, torchvision, ultralytics" >nul 2>&1
if not errorlevel 1 goto start

:install
call "%~dp0Install-Web.bat" --no-pause
if errorlevel 1 goto failed

:start
"%VENV_PY%" web_app.py --model "%HALPE_WEIGHTS%" %*
if errorlevel 1 goto failed
exit /b 0

:missing_weights
echo Halpe-26 weights are missing: %HALPE_WEIGHTS%
echo Extract the entire RightMove archive before launching.
goto failed

:failed
echo RightMove could not start. See the error above.
pause
exit /b 1

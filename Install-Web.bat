@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto dependencies
py -3.11 -c "import struct; assert struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto system_python
py -3.11 -m venv .venv
if errorlevel 1 goto failed
goto dependencies

:system_python
python -c "import sys, struct; assert sys.version_info[:2] in ((3, 11), (3, 12)) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv .venv
if errorlevel 1 goto failed

:dependencies
echo Installing PyTorch and torchvision...
"%VENV_PY%" -m pip install --disable-pip-version-check --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
if errorlevel 1 goto failed
echo Installing RightMove dependencies...
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements-web.txt torch==2.14.1 torchvision==0.29.1
if errorlevel 1 goto failed
"%VENV_PY%" -c "import cv2, numpy, torch, torchvision, ultralytics"
if errorlevel 1 goto failed
echo Installation complete. Run Start-Web-Halpe26.bat.
if /i not "%~1"=="--no-pause" pause
exit /b 0

:missing_python
echo Install 64-bit Python 3.11 or 3.12 and enable Add Python to PATH.
goto failed

:failed
echo Installation failed. See the error above.
if /i not "%~1"=="--no-pause" pause
exit /b 1

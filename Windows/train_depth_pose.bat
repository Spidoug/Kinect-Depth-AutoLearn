@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PTR=%~dp0runtime_python.txt"
if not exist "%PTR%" set "PTR=%LOCALAPPDATA%\KinectDepthAutoLearn\runtime_python.txt"
if not exist "%PTR%" (
  echo [ERROR] Runtime not found. Run install_autolearn.bat first.
  exit /b 20
)
set /p PY=<"%PTR%"
if not defined PY exit /b 21
if not exist "%PY%" (
  echo [ERROR] Configured Python does not exist: %PY%
  echo Re-run install_autolearn.bat to repair the persistent runtime pointer.
  exit /b 22
)
echo ============================================================
echo Kinect Depth AutoLearn - trainer
echo ============================================================
if "%~1"=="" (echo Dataset: latest valid session will be selected automatically.) else (echo Dataset: %~1)
echo Python: %PY%
echo Checkpoints: every 100 batches + safe pause checkpoint
echo Resume: automatic from output\last.pt
echo Console: progress / elapsed time / ETA every 15 seconds
echo ============================================================
echo.
"%PY%" "%~dp0trainer\train_depth_pose.py" %* --resume auto --checkpoint-batches 100 --console-seconds 15
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Training ended with error code %RC%.
exit /b %RC%

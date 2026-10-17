@echo off
setlocal
cd /d "%~dp0"
set "PTR=%~dp0runtime_python.txt"
if not exist "%PTR%" set "PTR=%LOCALAPPDATA%\KinectDepthAutoLearn\runtime_python.txt"
if not exist "%PTR%" (echo [ERROR] Runtime not found. Run install_autolearn.bat first.& exit /b 20)
set /p PY=<"%PTR%"
if not exist "%PY%" (echo [ERROR] Configured Python does not exist: %PY%& exit /b 22)
"%PY%" "%~dp0trainer\training_control.py" status %*
exit /b %ERRORLEVEL%

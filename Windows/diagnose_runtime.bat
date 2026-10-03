@echo off
setlocal
cd /d "%~dp0"
title Kinect Depth AutoLearn - Diagnostics

echo ============================================================
echo  Kinect Depth AutoLearn - runtime diagnostics
echo ============================================================
echo.

set "PTR=%~dp0runtime_python.txt"
if not exist "%PTR%" (
  echo [ERROR] runtime_python.txt does not exist.
  echo Run install_autolearn.bat.
  goto :end
)

set /p PY=<"%PTR%"
if not defined PY (
  echo [ERROR] runtime_python.txt is empty.
  goto :end
)
if not exist "%PY%" (
  echo [ERROR] The recorded Python executable does not exist:
  echo   "%PY%"
  echo Run install_autolearn.bat again.
  goto :end
)

"%PY%" "%~dp0trainer\diagnose_runtime.py"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo [OK] Runtime functional.
) else (
  echo [ERROR] Diagnostics ended with code %RC%.
)

:end
echo.
pause

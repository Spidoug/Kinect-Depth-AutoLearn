@echo off
setlocal
cd /d "%~dp0"
title Kinect Depth AutoLearn - Installation

echo ============================================================
echo  Kinect Depth AutoLearn - installation
echo ============================================================
echo.

if "%~1"=="" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autolearn.ps1"
) else (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autolearn.ps1" -Backend "%~1"
)
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Installation complete.
) else (
  echo [ERROR] Installation ended with code %RC%.
  echo See "%~dp0install_autolearn.log"
)
echo.
pause
exit /b %RC%

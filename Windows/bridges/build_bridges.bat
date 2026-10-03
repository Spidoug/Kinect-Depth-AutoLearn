@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Kinect Depth AutoLearn - optional bridge build
echo ============================================================
echo.
where cl.exe >nul 2>&1 || (
  echo [ERROR] cl.exe was not found.
  echo Open "x64 Native Tools Command Prompt for VS" and run this file again.
  pause & exit /b 1
)
if not exist bin mkdir bin
if not exist src\BridgeProtocol.h (echo [ERROR] bridge sources are missing.& pause & exit /b 1)

echo [1/3] Microsoft Kinect SDK 1.8 / Kinect 360...
if defined KINECTSDK10_DIR (
  cl /nologo /EHsc /std:c++17 /I"%KINECTSDK10_DIR%inc" src\microsoft_kinect_v1_bridge.cpp /link /LIBPATH:"%KINECTSDK10_DIR%lib\amd64" Kinect10.lib /OUT:bin\microsoft_kinect_v1_bridge.exe
  if errorlevel 1 echo [WARNING] Kinect SDK 1.8 bridge failed to build. Verify the x64 Kinect for Windows SDK 1.8 installation.
) else (
  echo [WARNING] KINECTSDK10_DIR is not defined. Install Kinect for Windows SDK 1.8 to build this bridge.
)

echo.
echo [2/3] OpenKinect / libfreenect...
if defined LIBFREENECT_DIR (
  cl /nologo /EHsc /std:c++17 /I"%LIBFREENECT_DIR%\include" src\openkinect_bridge.cpp /link /LIBPATH:"%LIBFREENECT_DIR%\lib" freenect_sync.lib freenect.lib /OUT:bin\openkinect_bridge.exe
  if errorlevel 1 echo [WARNING] OpenKinect bridge failed to build. Verify LIBFREENECT_DIR and x64 libraries.
) else (
  echo [WARNING] LIBFREENECT_DIR is not defined. Point it to an x64 libfreenect build.
)

echo.
echo [3/3] Microsoft Kinect SDK 2.0 / Kinect v2...
if defined KINECTSDK20_DIR (
  cl /nologo /EHsc /std:c++17 /I"%KINECTSDK20_DIR%inc" src\microsoft_kinect_v2_bridge.cpp /link /LIBPATH:"%KINECTSDK20_DIR%Lib\x64" Kinect20.lib /OUT:bin\microsoft_kinect_v2_bridge.exe
  if errorlevel 1 echo [WARNING] Kinect SDK 2.0 bridge failed to build. Verify the x64 Kinect for Windows SDK 2.0 installation.
) else (
  echo [WARNING] KINECTSDK20_DIR is not defined. Install Kinect for Windows SDK 2.0 to build this bridge.
)

echo.
echo Done. Successfully built executables are in bridges\bin\
pause

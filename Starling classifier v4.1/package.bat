@echo off
REM ============================================================
REM Package the built dist\ folder into a single ZIP for sharing
REM Produces: StarlingClassifier_<date>.zip in the project folder
REM ============================================================

set "APP=%~dp0"
set "DIST=%APP%dist\StarlingClassifier"

if not exist "%DIST%" (
    echo ERROR: No build found at %DIST%
    echo Run build.bat first.
    pause & exit /b 1
)

REM Date stamp YYYY-MM-DD
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "STAMP=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%"

set "ZIP=%APP%StarlingClassifier_%STAMP%.zip"

if exist "%ZIP%" del "%ZIP%"

echo.
echo === Copying README into dist folder ===
if exist "%APP%dist_README.txt" copy /Y "%APP%dist_README.txt" "%DIST%\README.txt" >nul

echo === Zipping %DIST% ===
echo Output: %ZIP%
echo (this may take a couple of minutes for a 200MB folder)
echo.

REM PowerShell Compress-Archive is built into Windows 10/11
powershell -NoProfile -Command "Compress-Archive -Path '%DIST%\*' -DestinationPath '%ZIP%' -CompressionLevel Optimal"

if errorlevel 1 (
    echo ZIP creation failed.
    pause & exit /b 1
)

echo.
echo ============================================================
echo PACKAGE READY: %ZIP%
echo Send this file to recipients.
echo ============================================================
pause

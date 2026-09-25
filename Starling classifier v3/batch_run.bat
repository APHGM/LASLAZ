@echo off
REM ============================================================
REM Batch-process all LAS/LAZ files in a folder
REM ============================================================

set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=P:\COMMON\Arun\arunpy\Scripts\python.exe"
set "SCRIPT=%~dp0batch_process.py"

if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    pause & exit /b 1
)

REM ---- Input folder ------------------------------------------
set /p INPUT_FOLDER=Enter folder containing LAS/LAZ files:
if not exist "%INPUT_FOLDER%" (
    echo ERROR: Folder not found: %INPUT_FOLDER%
    pause & exit /b 1
)

REM ---- Tile size ---------------------------------------------
set TILE_SIZE=100
set /p TILE_SIZE=Tile size in metres [default 100]:

REM ---- Ground method -----------------------------------------
set GROUND_METHOD=csf
set /p GROUND_METHOD=Ground method  csf / grid  [default csf]:

REM ---- Output format -----------------------------------------
set OUT_FORMAT=laz
set /p OUT_FORMAT=Output format  laz / las  [default laz]:

echo.
echo ============================================================
echo Batch processing all LAS/LAZ files in:
echo   %INPUT_FOLDER%
echo Tile size : %TILE_SIZE% m
echo Ground    : %GROUND_METHOD%
echo Format    : %OUT_FORMAT%
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" "%INPUT_FOLDER%" ^
    --tile-size %TILE_SIZE% ^
    --ground-method %GROUND_METHOD% ^
    --output-format %OUT_FORMAT%

if errorlevel 1 (
    echo.
    echo BATCH FAILED.
    pause & exit /b 1
)

echo.
echo ============================================================
echo BATCH COMPLETE.  See each ^<stem^>_output\ folder for results.
echo ============================================================
pause

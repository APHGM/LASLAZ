@echo off
REM ============================================================
REM Starling Classifier — tile a single LAZ then launch the GUI
REM Folder-portable: uses %~dp0 (the folder this bat sits in)
REM ============================================================

set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=P:\COMMON\Arun\arunpy\Scripts\python.exe"
set "APP=%~dp0"
set "TILER=%APP%tile_single_laz.py"
set "GUI=%APP%main.py"

REM ---- Sanity checks -----------------------------------------
if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    pause & exit /b 1
)
if not exist "%TILER%" (
    echo ERROR: Tiler script not found: %TILER%
    pause & exit /b 1
)
if not exist "%GUI%" (
    echo ERROR: GUI script not found: %GUI%
    pause & exit /b 1
)

REM ---- Ask for input LAZ -------------------------------------
set /p INPUT_LAZ=Enter full path to source LAZ file:
if not exist "%INPUT_LAZ%" (
    echo ERROR: File not found: %INPUT_LAZ%
    pause & exit /b 1
)

REM ---- Ask for tile size (default 100) -----------------------
set TILE_SIZE=100
set /p TILE_SIZE=Tile size in metres [default 100]:

REM ---- Derive tile folder next to the input file -------------
for %%F in ("%INPUT_LAZ%") do (
    set "BASENAME=%%~nF"
    set "INPUT_DIR=%%~dpF"
)
set "TILE_DIR=%INPUT_DIR%Tile_%BASENAME%"
set /p TILE_DIR=Tile output folder [default %TILE_DIR%]:

echo.
echo ============================================================
echo Step 1/2  Tiling
echo   Input    %INPUT_LAZ%
echo   Output   %TILE_DIR%
echo   Tile     %TILE_SIZE% m
echo ============================================================
"%PYTHON%" "%TILER%" "%INPUT_LAZ%" "%TILE_DIR%" --tile %TILE_SIZE% --chunk 5000000
if errorlevel 1 (
    echo Tiling failed.
    pause & exit /b 1
)

echo.
echo ============================================================
echo Step 2/2  Launching GUI
echo ============================================================
"%PYTHON%" "%GUI%"

pause

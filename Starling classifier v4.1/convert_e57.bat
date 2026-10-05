@echo off
REM ============================================================
REM  E57 → LAZ Batch Converter
REM  Converts all E57 files in a folder to LAZ (alongside each E57)
REM  or a single E57 file to LAZ.
REM ============================================================

set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=P:\COMMON\Arun\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: Python not found.
    pause & exit /b 1
)

set "SCRIPT=%~dp0convert_e57.py"

REM ---- Input -------------------------------------------------
set /p INPUT=Enter E57 file or folder path:
if not exist "%INPUT%" (
    echo ERROR: Path not found: %INPUT%
    pause & exit /b 1
)

REM ---- Output ------------------------------------------------
echo.
echo Output options:
echo   1  Alongside each E57 (default)
echo   2  Custom output folder
set /p OUTCHOICE=Choose [1/2, default=1]:
if "%OUTCHOICE%"=="2" (
    set /p OUTDIR=Enter output folder path:
    set "OUT_ARG=--out "%OUTDIR%""
) else (
    set "OUT_ARG="
)

REM ---- Recursive ---------------------------------------------
set /p RECURSIVE=Search subfolders? [y/N]:
if /i "%RECURSIVE%"=="y" (
    set "REC_ARG=--recursive"
) else (
    set "REC_ARG="
)

echo.
echo ============================================================
echo Converting E57 files...
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" "%INPUT%" %OUT_ARG% %REC_ARG%

if errorlevel 1 (
    echo.
    echo CONVERSION FAILED — check errors above.
    pause & exit /b 1
)

echo.
echo ============================================================
echo CONVERSION COMPLETE.
echo ============================================================
pause

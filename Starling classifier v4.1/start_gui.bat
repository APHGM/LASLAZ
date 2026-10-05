@echo off
REM Launch the Starling Classifier GUI (prefers local venv, falls back to network)
set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=P:\COMMON\Arun\arunpy\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python not found. Tried:
    echo   D:\VSCode_Working\Python\arunpy\Scripts\python.exe
    echo   P:\COMMON\Arun\arunpy\Scripts\python.exe
    pause & exit /b 1
)

"%PYTHON%" "%~dp0main.py"
pause

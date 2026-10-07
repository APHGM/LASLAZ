@echo off
REM ============================================================
REM Clean PyInstaller build of Starling Classifier
REM Removes previous dist/build/cache then rebuilds from build.spec
REM ============================================================

REM Prefer local venv (D:); fall back to network venv (P:) if not found
set "PYTHON=D:\VSCode_Working\Python\arunpy\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=P:\COMMON\Arun\arunpy\Scripts\python.exe"
set "APP=%~dp0"

if not exist "%PYTHON%" (
    echo ERROR: Python not found. Tried:
    echo   D:\VSCode_Working\Python\arunpy\Scripts\python.exe
    echo   P:\COMMON\Arun\arunpy\Scripts\python.exe
    pause & exit /b 1
)
if not exist "%APP%build.spec" (
    echo ERROR: build.spec not found in %APP%
    pause & exit /b 1
)

echo.
echo === Cleaning previous build artefacts ===
if exist "%APP%dist"          rmdir /s /q "%APP%dist"
if exist "%APP%build"         rmdir /s /q "%APP%build"
if exist "%APP%__pycache__"   rmdir /s /q "%APP%__pycache__"
for /d /r "%APP%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)

echo.
echo === Building from build.spec ===
"%PYTHON%" -m PyInstaller --noconfirm --clean "%APP%build.spec"

if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    pause & exit /b 1
)

echo.
echo ============================================================
echo BUILD COMPLETE.
echo Output: %APP%dist\StarlingClassifier_v4.2\StarlingClassifier_v4.2.exe
echo ============================================================
pause

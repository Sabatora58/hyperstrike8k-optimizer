@echo off
title HYPERSTRIKE 8K Optimizer - Fix GPU (DirectML)
cd /d "%~dp0"
set PY=
if exist runtime\python.exe set PY=runtime\python.exe
if "%PY%"=="" if exist buildenv\tools\python.exe set PY=buildenv\tools\python.exe
if "%PY%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set PY=python
)
if "%PY%"=="" (
    echo No Python found. Run setup_portable.bat first.
    pause
    exit /b 1
)
echo Using Python: %PY%
echo Removing conflicting CPU onnxruntime and reinstalling DirectML build...
%PY% -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
%PY% -m pip install --no-warn-script-location --force-reinstall onnxruntime-directml
if errorlevel 1 (
    echo [ERROR] Reinstall failed. Check your internet connection.
    pause
    exit /b 1
)
%PY% -c "import onnxruntime as o; print('available providers:', o.get_available_providers())"
echo.
echo Done. Restart the app and check the header shows DmlExecutionProvider (GPU).
pause

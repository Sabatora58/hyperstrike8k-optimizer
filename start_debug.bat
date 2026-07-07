@echo off
title HYPERSTRIKE 8K Optimizer - DEBUG (visible console)
cd /d "%~dp0"
set PY=
if exist runtime\python.exe set PY=runtime\python.exe
if "%PY%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set PY=python
)
if "%PY%"=="" (
    echo No Python found. Run setup_portable.bat first, or copy the
    echo "runtime" folder from your previous version folder.
    pause
    exit /b 1
)
echo Running with visible console for debugging. Errors will stay on screen.
%PY% app.py
echo.
echo ---- exited ----
pause

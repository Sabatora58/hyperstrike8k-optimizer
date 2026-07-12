@echo off
title HYPERSTRIKE 8K Optimizer - Analyze external match recording
cd /d "%~dp0"
echo ==================================================================
echo  Analyze an external match recording (AimSync Recorder etc.)
echo  Drag the recording FOLDER onto this .bat, or run it and paste
echo  the folder path when asked. The folder must contain
echo  input_log.jsonl (required) and apex_record.mp4 (optional).
echo ==================================================================
echo.

set PY=
if exist runtime\python.exe set PY=runtime\python.exe
if "%PY%"=="" if exist buildenv\tools\python.exe set PY=buildenv\tools\python.exe
if "%PY%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set PY=python
)
if "%PY%"=="" (
    echo [ERROR] No Python found. Run setup_portable.bat first.
    pause
    exit /b 1
)

set FOLDER=%~1
if "%FOLDER%"=="" set /p FOLDER=Recording folder path:
if "%FOLDER%"=="" (
    echo [ERROR] No folder given.
    pause
    exit /b 1
)

%PY% ingest_recording.py "%FOLDER%"
echo.
pause
exit /b 0

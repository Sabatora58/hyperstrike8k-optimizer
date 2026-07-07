@echo off
title HYPERSTRIKE 8K Local Optimizer (Portable)
cd /d "%~dp0"

if exist runtime\pythonw.exe goto LAUNCH
if exist runtime\python.exe goto VISIBLE
echo ------------------------------------------------------------
echo runtime folder not found in THIS folder.
echo   Option A: run setup_portable.bat (downloads everything)
echo   Option B: copy the "runtime" and "models" folders from
echo             your previous version folder into this folder.
echo ------------------------------------------------------------
pause
exit /b 1

:LAUNCH
start "" runtime\pythonw.exe app.py
echo Started in background. Checking that the server is up...
timeout /t 6 >nul

rem Health check on ports 8720-8724 (auto port fallback range)
set OK=
for %%p in (8720 8721 8722 8723 8724) do (
    if not defined OK (
        curl -s -o nul --max-time 2 http://127.0.0.1:%%p/api/status && set OK=%%p
    )
)
if defined OK (
    echo Server is running on port %OK%. Your browser should open automatically.
    echo To stop the app, use the "quit" button in the web UI.
    timeout /t 3 >nul
    exit /b 0
)

echo.
echo [WARNING] The app did not start in background mode.
echo Recent log (hyperstrike.log):
echo ------------------------------------------------------------
if exist hyperstrike.log (
    powershell -NoProfile -Command "Get-Content hyperstrike.log -Tail 25"
) else (
    echo (no log file found)
)
echo ------------------------------------------------------------
echo Restarting in VISIBLE mode so you can see the error:
echo.

:VISIBLE
runtime\python.exe app.py
echo.
echo App stopped. If it crashed, the error is shown above.
pause

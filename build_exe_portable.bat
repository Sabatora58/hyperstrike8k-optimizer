@echo off
title HYPERSTRIKE 8K Optimizer - Build EXE (no Python install needed)
cd /d "%~dp0"
echo ==================================================================
echo  Builds HyperStrike8K-Optimizer.exe WITHOUT installing Python.
echo  A private full Python (NuGet package) is downloaded into the
echo  "buildenv" folder. Nothing is installed on your system.
echo  Internet is required for the first run only.
echo ==================================================================
echo.

if exist buildenv\tools\python.exe goto HAVEPY

echo [1/4] Downloading portable full Python 3.11 (about 16 MB)...
curl -L -o py-full.zip https://www.nuget.org/api/v2/package/python/3.11.9
if errorlevel 1 goto DLFAIL

echo [2/4] Extracting to buildenv...
mkdir buildenv
tar -xf py-full.zip -C buildenv
del py-full.zip
if not exist buildenv\tools\python.exe goto EXFAIL

echo [3/4] Enabling pip...
buildenv\tools\python.exe -m ensurepip --upgrade
if errorlevel 1 goto PIPFAIL

:HAVEPY
echo [4/4] Installing build dependencies into buildenv...
buildenv\tools\python.exe -m pip install --no-warn-script-location -U pip
buildenv\tools\python.exe -m pip install --no-warn-script-location fastapi uvicorn numpy mss pygame opencv-python winocr onnxruntime-directml pyinstaller
if errorlevel 1 goto PIPFAIL

echo.
echo Building EXE (this may take several minutes)...
buildenv\tools\python.exe -m PyInstaller --noconfirm --onefile --noconsole --name HyperStrike8K-Optimizer --add-data "static;static" --collect-all onnxruntime --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on app.py
if errorlevel 1 goto FAIL

if not exist dist\models mkdir dist\models
if exist models\yolov8n.onnx copy /y models\yolov8n.onnx dist\models\ >nul
if not exist models\yolov8n.onnx echo NOTE: models\yolov8n.onnx not found - run setup_portable.bat first if you want AI vision in the EXE.

echo.
echo Done: dist\HyperStrike8K-Optimizer.exe  (no console window, no Python needed to run)
echo Keep dist\models\yolov8n.onnx next to the EXE when distributing.
echo Runtime logs: hyperstrike.log next to the EXE.
pause
exit /b 0

:DLFAIL
echo [ERROR] Download failed. Check your internet connection.
pause
exit /b 1

:EXFAIL
echo [ERROR] Extraction failed - buildenv\tools\python.exe not found.
pause
exit /b 1

:PIPFAIL
echo [ERROR] Package installation failed. See the error above.
pause
exit /b 1

:FAIL
echo Build failed. See the error above.
pause
exit /b 1

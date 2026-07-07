@echo off
title HYPERSTRIKE 8K Optimizer - Portable Setup (no Python install needed)
cd /d "%~dp0"
echo ============================================================
echo  Portable setup: downloads a private embedded Python into
echo  the "runtime" folder. Nothing is installed on your system.
echo  Internet is required for this first-time setup only.
echo ============================================================
echo.

if exist runtime\python.exe goto HAVEPY

echo [1/5] Downloading embedded Python 3.11 (about 11 MB)...
curl -L -o py-embed.zip https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
if errorlevel 1 goto DLFAIL

echo [2/5] Extracting...
mkdir runtime
tar -xf py-embed.zip -C runtime
del py-embed.zip

echo [3/5] Enabling package support...
powershell -NoProfile -Command "(Get-Content runtime\python311._pth) -replace '#import site','import site' | Set-Content runtime\python311._pth"

echo [4/5] Installing pip...
curl -L -o runtime\get-pip.py https://bootstrap.pypa.io/get-pip.py
runtime\python.exe runtime\get-pip.py --no-warn-script-location
if errorlevel 1 goto PIPFAIL

:HAVEPY
echo [5/5] Installing packages into the runtime folder...
runtime\python.exe -m pip install --no-warn-script-location fastapi uvicorn numpy mss pygame opencv-python winocr
if errorlevel 1 goto PIPFAIL

:MODEL
if exist models\yolov8n.onnx goto GPUSTEP
echo Preparing enemy-detection model (YOLOv8n)...
echo (NOTE: this step may pull in a CPU onnxruntime - it is replaced below)
runtime\python.exe -m pip install --no-warn-script-location ultralytics
runtime\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
if not exist models mkdir models
if exist yolov8n.onnx move /y yolov8n.onnx models\ >nul

:GPUSTEP
echo Installing GPU runtime LAST so nothing overwrites it.
echo DirectML works on NVIDIA/AMD/Intel with no CUDA/cuDNN setup.
runtime\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
runtime\python.exe -m pip install --no-warn-script-location --force-reinstall onnxruntime-directml
runtime\python.exe -c "import onnxruntime as o; print('available providers:', o.get_available_providers())"

:DONE
echo.
echo Portable setup complete. Run start_portable.bat to launch.
pause
exit /b 0

:DLFAIL
echo [ERROR] Download failed. Check your internet connection.
pause
exit /b 1

:PIPFAIL
echo [ERROR] Package installation failed. See the error above.
pause
exit /b 1

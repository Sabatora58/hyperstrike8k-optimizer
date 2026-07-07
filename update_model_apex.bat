@echo off
title HYPERSTRIKE 8K Optimizer - Install Apex-trained model
cd /d "%~dp0"
echo ==================================================================
echo  Installs the community Apex-trained YOLOv8 model (apex_v8n)
echo  with Enemy/Teammate IFF. Source:
echo  github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF (MIT-forked)
echo  Only the detection weight is used, for OFFLINE analysis only.
echo ==================================================================
echo.

set PY=
if exist buildenv\tools\python.exe set PY=buildenv\tools\python.exe
if "%PY%"=="" if exist runtime\python.exe set PY=runtime\python.exe
if "%PY%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set PY=python
)
if "%PY%"=="" (
    echo [ERROR] No Python found. Run setup_portable.bat or build_exe_portable.bat first.
    pause
    exit /b 1
)
echo Using Python: %PY%

echo [1/4] Downloading repository archive...
curl -L -o apex-repo.zip https://codeload.github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF/zip/refs/heads/master
if errorlevel 1 goto DLFAIL

echo [2/4] Extracting weight file...
if exist _apexrepo rmdir /s /q _apexrepo
mkdir _apexrepo
tar -xf apex-repo.zip -C _apexrepo
del apex-repo.zip

echo [3/4] Converting .pt to ONNX (installs ultralytics if needed)...
%PY% -m pip install --no-warn-script-location -q ultralytics
%PY% -c "import glob,sys; from ultralytics import YOLO; pts=glob.glob('_apexrepo/**/*.pt', recursive=True); sys.exit('no .pt found') if not pts else None; print('weight:', pts[0]); YOLO(pts[0]).export(format='onnx', imgsz=640)"
if errorlevel 1 goto CONVFAIL

echo [4/4] Installing to models\apex.onnx ...
if not exist models mkdir models
for /r _apexrepo %%f in (*.onnx) do copy /y "%%f" models\apex.onnx >nul
if not exist models\apex.onnx goto CONVFAIL
rmdir /s /q _apexrepo

echo Restoring DirectML runtime (conversion may have installed CPU onnxruntime)...
%PY% -m pip uninstall -y onnxruntime onnxruntime-gpu >nul 2>nul
%PY% -m pip install --no-warn-script-location --force-reinstall onnxruntime-directml >nul

echo.
echo Done. models\apex.onnx installed.
echo The app now auto-detects it (header shows "apex.onnx / Apex IFF").
echo Enemy-only detection: teammates are excluded from analysis.
echo If you build an EXE, copy models\apex.onnx into dist\models\ as well.
pause
exit /b 0

:DLFAIL
echo [ERROR] Download failed. Check your internet connection.
pause
exit /b 1

:CONVFAIL
echo [ERROR] Conversion failed. See the error above.
if exist _apexrepo rmdir /s /q _apexrepo
pause
exit /b 1

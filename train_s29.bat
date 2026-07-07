@echo off
title HYPERSTRIKE - Train Season 29 model (legends + training dummies)
cd /d "%~dp0"
echo ==================================================================
echo  Fine-tunes the Apex IFF model on YOUR Season 29 screenshots
echo  (dataset\raw) so it covers current legends and training dummies.
echo  Steps: pseudo-label - train - export ONNX - install as apex.onnx
echo  NOTE: training needs a NVIDIA GPU for reasonable speed.
echo        CPU training works but may take hours.
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

echo [1/6] Preparing dataset (pseudo-labels from current model)...
%PY% prep_dataset.py
if errorlevel 1 goto FAIL

echo [2/6] Installing training packages (ultralytics + PyTorch)...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo   No NVIDIA GPU detected - installing CPU PyTorch (training will be SLOW)
    %PY% -m pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo   NVIDIA GPU detected - installing CUDA PyTorch
    %PY% -m pip install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
%PY% -m pip install --no-warn-script-location ultralytics
if errorlevel 1 goto FAIL

echo [3/6] Getting pretrained weight apex_v8n.pt (if missing)...
if exist models\apex_v8n.pt goto TRAIN
curl -L -o apex-repo.zip https://codeload.github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF/zip/refs/heads/master
if errorlevel 1 goto FAIL
if exist _apexrepo rmdir /s /q _apexrepo
mkdir _apexrepo
tar -xf apex-repo.zip -C _apexrepo
del apex-repo.zip
if not exist models mkdir models
for /r _apexrepo %%f in (*.pt) do copy /y "%%f" models\apex_v8n.pt >nul
rmdir /s /q _apexrepo
if not exist models\apex_v8n.pt goto FAIL

:TRAIN
echo [4/6] Training (50 epochs, this is the long step)...
%PY% -m ultralytics.cfg train model=models\apex_v8n.pt data=dataset\apex_s29.yaml epochs=50 imgsz=640 batch=-1 project=dataset\runs name=s29 exist_ok=True 2>nul
if errorlevel 1 (
    %PY% -c "from ultralytics import YOLO; YOLO('models/apex_v8n.pt').train(data='dataset/apex_s29.yaml', epochs=50, imgsz=640, batch=-1, project='dataset/runs', name='s29', exist_ok=True)"
)
if not exist dataset\runs\s29\weights\best.pt goto FAIL

echo [5/6] Exporting to ONNX and installing...
%PY% -c "from ultralytics import YOLO; YOLO('dataset/runs/s29/weights/best.pt').export(format='onnx', imgsz=640)"
if exist models\apex.onnx copy /y models\apex.onnx models\apex_backup.onnx >nul
copy /y dataset\runs\s29\weights\best.onnx models\apex.onnx >nul
if not exist models\apex.onnx goto FAIL

echo [6/6] Restoring DirectML inference runtime...
%PY% -m pip uninstall -y onnxruntime onnxruntime-gpu >nul 2>nul
%PY% -m pip install --no-warn-script-location --force-reinstall onnxruntime-directml >nul

echo.
echo Done. Season 29 model installed as models\apex.onnx
echo (previous model saved as models\apex_backup.onnx)
echo Validation metrics: dataset\runs\s29\results.png
echo Restart the app - header should show "apex.onnx".
pause
exit /b 0

:FAIL
echo [ERROR] See the message above. Common causes:
echo  - dataset\raw is empty (enable dataset capture in the app and play first)
echo  - out of GPU memory (edit this bat: batch=-1 to batch=8)
pause
exit /b 1

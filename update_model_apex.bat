@echo off
title HYPERSTRIKE 8K Optimizer - Install Apex-trained model
cd /d "%~dp0"
echo ==================================================================
echo  Installs the community Apex-trained YOLOv8 model with
echo  Enemy/Teammate IFF (2 classes: 0=Teammate, 1=Enemy). Source:
echo  github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF
echo  Only the detection weight is used, for OFFLINE analysis only.
echo ==================================================================
echo.

echo [1/3] Downloading repository archive...
curl -L -o apex-repo.zip https://codeload.github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF/zip/refs/heads/master
if errorlevel 1 goto DLFAIL

echo [2/3] Extracting...
if exist _apexrepo rmdir /s /q _apexrepo
mkdir _apexrepo
tar -xf apex-repo.zip -C _apexrepo
del apex-repo.zip

echo [3/3] Installing bundled ONNX weight (no conversion needed)...
if not exist models mkdir models
rem Prefer the pre-converted 2-class IFF weight shipped in the repo.
rem This avoids loading .pt (pickle) files and does not touch pip,
rem so the DirectML onnxruntime stays intact.
set FOUND=
for /r _apexrepo %%f in (apex_7w_8n.onnx) do (
    copy /y "%%f" models\apex.onnx >nul
    set FOUND=1
)
if not defined FOUND (
    echo [ERROR] apex_7w_8n.onnx not found in the repository archive.
    echo         The repo layout may have changed. Keeping current model.
    rmdir /s /q _apexrepo
    pause
    exit /b 1
)
rmdir /s /q _apexrepo

echo.
echo Done. models\apex.onnx installed (fp16, 2-class IFF).
echo The app auto-detects it (header shows "apex.onnx").
echo Enemy-only detection: teammates are excluded from analysis.
echo If you build an EXE, copy models\apex.onnx into dist\models\ as well.
pause
exit /b 0

:DLFAIL
echo [ERROR] Download failed. Check your internet connection.
pause
exit /b 1

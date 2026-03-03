@echo off
title HealthFusion AI — Setup & Launch
color 0A

echo.
echo  ╔════════════════════════════════════════════╗
echo  ║     HealthFusion AI — Auto Setup          ║
echo  ║     FastAPI + SQLite Backend               ║
echo  ╚════════════════════════════════════════════╝
echo.

:: Check Python
python --version 2>NUL
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Install from: https://python.org/downloads
    echo  Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo  [1/4] Python found. Installing packages...
echo.

pip install fastapi==0.111.0 uvicorn[standard]==0.30.1 python-multipart==0.0.9 ^
    numpy pandas scikit-learn xgboost reportlab bcrypt pillow --quiet

echo.
echo  [2/4] Packages installed!
echo.

:: Optional packages
echo  [3/4] Installing optional packages (SHAP + PyTorch)...
pip install shap --quiet 2>NUL && echo  [OK] SHAP installed || echo  [SKIP] SHAP failed (optional)
pip install torch torchvision --quiet 2>NUL && echo  [OK] PyTorch installed || echo  [SKIP] PyTorch failed (optional - needed for skin detection)

echo.
echo  [4/4] Starting HealthFusion AI server...
echo.
echo  ╔════════════════════════════════════════════╗
echo  ║  Open browser: http://localhost:8000       ║
echo  ║  Press Ctrl+C to stop server              ║
echo  ╚════════════════════════════════════════════╝
echo.

:: Go to script directory
cd /d "%~dp0"

:: Start server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause

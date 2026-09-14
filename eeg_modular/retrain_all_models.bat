@echo off
chcp 65001 >nul
REM 批量重新训练所有模型（使用 NPY 数据）

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
cd /d "%~dp0"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found: %PYTHON_EXE%
    echo Please create the local virtual environment at:
    echo         %~dp0.venv
    echo Expected interpreter: .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo ==========================================
echo 开始重新训练所有模型...
echo ==========================================

REM 设置 Python 环境
set "PYTHONPATH=%PYTHONPATH%;%CD%"

REM 训练 SVM
echo.
echo 训练 SVM 模型...
"%PYTHON_EXE%" -m scripts.train -c configs/svm.yaml

REM 训练 MLP
echo.
echo 训练 MLP 模型...
"%PYTHON_EXE%" -m scripts.train -c configs/mlp.yaml

REM 训练 RF
echo.
echo 训练 RF 模型...
"%PYTHON_EXE%" -m scripts.train -c configs/rf.yaml

REM 训练 XGBoost
echo.
echo 训练 XGBoost 模型...
"%PYTHON_EXE%" -m scripts.train -c configs/xgb.yaml

REM 训练 HYBRID
echo.
echo 训练 HYBRID 模型...
"%PYTHON_EXE%" -m scripts.train -c configs/hybrid.yaml

echo.
echo ==========================================
echo 所有模型训练完成！
echo ==========================================

REM 验证一致性
echo.
echo 验证测试集一致性...
"%PYTHON_EXE%" verify_consistency.py

pause

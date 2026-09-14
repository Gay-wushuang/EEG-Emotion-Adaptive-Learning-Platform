# PowerShell 脚本：使用项目本地虚拟环境重新训练所有模型

$ErrorActionPreference = "Continue"

# 所有路径均相对于本脚本，避免依赖调用时的当前目录或开发机盘符。
$working_dir = $PSScriptRoot
$python_exe = Join-Path $working_dir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python_exe -PathType Leaf)) {
    Write-Host "[ERROR] Python interpreter not found: $python_exe" -ForegroundColor Red
    Write-Host "Please create the local virtual environment at:" -ForegroundColor Yellow
    Write-Host "        $(Join-Path $working_dir '.venv')" -ForegroundColor Yellow
    Write-Host "Expected interpreter: .venv\Scripts\python.exe" -ForegroundColor Yellow
    exit 1
}

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$working_dir;$($env:PYTHONPATH)"
} else {
    $env:PYTHONPATH = $working_dir
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "开始重新训练所有模型（使用 NPY 数据）" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 切换到工作目录
Set-Location $working_dir

# 训练 SVM
Write-Host "训练 SVM 模型..." -ForegroundColor Yellow
& $python_exe -m scripts.train -c configs/svm.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "SVM 训练失败！" -ForegroundColor Red
} else {
    Write-Host "SVM 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 MLP
Write-Host "训练 MLP 模型..." -ForegroundColor Yellow
& $python_exe -m scripts.train -c configs/mlp.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "MLP 训练失败！" -ForegroundColor Red
} else {
    Write-Host "MLP 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 RF
Write-Host "训练 RF 模型..." -ForegroundColor Yellow
& $python_exe -m scripts.train -c configs/rf.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "RF 训练失败！" -ForegroundColor Red
} else {
    Write-Host "RF 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 XGBoost
Write-Host "训练 XGBoost 模型..." -ForegroundColor Yellow
& $python_exe -m scripts.train -c configs/xgb.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "XGBoost 训练失败！" -ForegroundColor Red
} else {
    Write-Host "XGBoost 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 HYBRID
Write-Host "训练 HYBRID 模型..." -ForegroundColor Yellow
& $python_exe -m scripts.train -c configs/hybrid.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "HYBRID 训练失败！" -ForegroundColor Red
} else {
    Write-Host "HYBRID 训练完成！" -ForegroundColor Green
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "所有模型训练完成！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 验证一致性
Write-Host "验证测试集一致性..." -ForegroundColor Yellow
& $python_exe verify_consistency.py
Write-Host ""

Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

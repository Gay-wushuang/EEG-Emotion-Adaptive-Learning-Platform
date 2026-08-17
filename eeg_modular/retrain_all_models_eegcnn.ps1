# PowerShell 脚本：使用 eegcnn 环境重新训练所有模型

$ErrorActionPreference = "Continue"

# 设置环境路径
$env:PYTHONPATH = "E:\ERE\Documents\Work\EEG_AI_AR\eeg_modular"
$eegcnn_python = "E:\anaconda3\envs\eegcnn\python.exe"
$working_dir = "E:\ERE\Documents\Work\EEG_AI_AR\eeg_modular"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "开始重新训练所有模型（使用 NPY 数据）" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 切换到工作目录
Set-Location $working_dir

# 训练 SVM
Write-Host "训练 SVM 模型..." -ForegroundColor Yellow
& $eegcnn_python -m scripts.train -c configs/svm.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "SVM 训练失败！" -ForegroundColor Red
} else {
    Write-Host "SVM 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 MLP
Write-Host "训练 MLP 模型..." -ForegroundColor Yellow
& $eegcnn_python -m scripts.train -c configs/mlp.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "MLP 训练失败！" -ForegroundColor Red
} else {
    Write-Host "MLP 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 RF
Write-Host "训练 RF 模型..." -ForegroundColor Yellow
& $eegcnn_python -m scripts.train -c configs/rf.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "RF 训练失败！" -ForegroundColor Red
} else {
    Write-Host "RF 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 XGBoost
Write-Host "训练 XGBoost 模型..." -ForegroundColor Yellow
& $eegcnn_python -m scripts.train -c configs/xgb.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "XGBoost 训练失败！" -ForegroundColor Red
} else {
    Write-Host "XGBoost 训练完成！" -ForegroundColor Green
}
Write-Host ""

# 训练 HYBRID
Write-Host "训练 HYBRID 模型..." -ForegroundColor Yellow
& $eegcnn_python -m scripts.train -c configs/hybrid.yaml
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
& $eegcnn_python verify_consistency.py
Write-Host ""

Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
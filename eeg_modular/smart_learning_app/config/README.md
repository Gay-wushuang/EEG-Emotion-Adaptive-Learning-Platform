# 产品配置说明

`application.yaml` 是桌面产品专用配置，不复用旧的
`realtime_inference/config/config.yaml`，避免误加载历史模型和Scaler。

- 生产资产以 `production_baseline_v1/` 为唯一权威来源。
- 内部类别顺序固定为 `happy / normal / sad`。
- 对外显示映射为 `positive / neutral / negative`。
- ATT/MED仅作辅助展示和反馈，不进入分类张量。
- 当前部署阈值遵循生产包内 `confidence_policy.json`：0.60、无温度缩放。
- 90.20%高置信度实验工作点没有直接部署到此配置。

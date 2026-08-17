# 七阶段诊断复测协议

## 原顺序复测

Raw采集预留30分钟，覆盖18分钟任务、自评填写和Connector启动延迟：

```powershell
python realtime_inference\tools\capture_mindwave_csv.py `
  --wait-for-raw-timeout 30 `
  --duration 1800 `
  --output realtime_inference\captures\diagnostic_repeat_original.csv
```

事件窗口：

```powershell
python scripts\mark_five_stage_experiment.py `
  --order original `
  --output realtime_inference\captures\diagnostic_repeat_original_events.csv
```

顺序：静息→正向→洗脱→普通→洗脱→受挫→恢复。

## 变序实验

```powershell
python scripts\mark_five_stage_experiment.py `
  --order changed `
  --output realtime_inference\captures\diagnostic_changed_order_events.csv
```

顺序：静息→普通→洗脱→正向→洗脱→受挫→恢复。

## 共同规则

- 自动干预关闭，不在实验过程中查看模型输出。
- 阶段结束记录愉悦、压力、专注、疲劳、主观类别及动作/眨眼/佩戴不适。
- 分析保存未拒识概率与logits，并逐窗口计算Poor Signal、Raw RMS、峰峰值、50 Hz能量占比、0.5–4 Hz低频占比和异常峰数。
- 跨阶段窗口、自评填写时段不进入阶段指标。
- 使用不重叠30秒块、阶段中位数或时间块bootstrap；实验场次才是独立重复。


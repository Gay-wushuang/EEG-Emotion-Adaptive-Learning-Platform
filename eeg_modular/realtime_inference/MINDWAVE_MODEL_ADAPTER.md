# MindWave 与现有情绪模型适配确认

## 已确认的模型规格

- 模型：`MultiModalCVAECNN`，类别顺序为 `happy / sad / normal`。
- 输入不是电极拓扑矩阵，而是字典中的四个模态：`filtered`、`powerspec`、`att`、`med`。
- 单样本每个模态形状为 `(10, 4)`；推理批次张量为 `(B, 10, 4)`，当前单次推理为 `(1, 10, 4)`。
- 四维特征顺序均为 `[mean, std, max, min]`。
- CVAE 展平输入总长为 `4 modalities × 10 × 4 = 160`。
- MindWave 是单通道设备，物理通道为额前 `FP1`，参考/地电极位于耳夹；当前模型没有多通道位置维。
- 工程配置采样率为 `512 Hz`。
- 对405段训练文件的核查显示：Raw EEG 每段中位数约15470点、ATT/MED每段中位数约30条，时长均约30秒。因此模型观察窗修正为 `30 s ≈ 15360 points`，再划分为10个时间段；实时推理步长为2秒，使用重叠窗口。

## 原始 CSV

每个 Raw EEG 样点占一行：

```text
timestamp_unix,timestamp_iso_utc,sample_index,raw_eeg,attention,meditation,poor_signal
```

ATT、MED、Poor Signal 的设备更新频率低于 Raw EEG，采集器将最近一次值向前填充到每个 512 Hz 样点。时间戳以第一个 Raw 包的接收时间为锚点，按 `sample_index / 512` 重建均匀采样时间；它是主机时间，不是设备硬件时钟。

启动 ThinkGear Connector 并连接头环后采集 10 秒：

```powershell
python tools/capture_mindwave_csv.py --duration 10 --output captures/mindwave_10s.csv
```

## CSV 到模型张量

1. 读取 CSV，按时间排序并对 ATT、MED、Poor Signal 前向填充。
2. 取30秒窗口；若不足则处于预热状态并拒绝推理。
3. 质量门控：窗口内至少 80% 样点满足 `poor_signal < 200`。
4. Raw EEG 去直流，并进行 1–45 Hz FFT 带通，得到 `filtered`。
5. 将 filtered 分成 10 段，每段计算 `[mean,std,max,min]`，得到 `(10,4)`。
6. 从 filtered 的短子窗计算 1–45 Hz 宽带对数功率包络，再按 10 段计算四项统计量，得到 `powerspec (10,4)`。
7. ATT、MED 各分为 10 段并计算相同四项统计量，得到各自 `(10,4)`。
8. 必须调用训练时保存的 `scaler_{modality}.joblib`，分别标准化四个模态。
9. 增加 batch 维，输出四个 `float32 (1,10,4)` 张量。

转换命令：

```powershell
python tools/mindwave_to_tensor.py captures/mindwave_10s.csv `
  --scalers-dir ../features `
  --output captures/mindwave_input.npz
```

## 已发现的部署风险

正式配置已改为 `skip_scaling: false`，并明确指定 `../features`。启动时任何 scaler 缺失或输入维数不是4都会直接报错。

训练预处理会先按100 ms重采样，再执行线性插值、前向填充和后向填充；ATT/MED原始频率约1 Hz。旧实时代码对 `att`、`med` 写入 `[value, 0, value, value]`，不等于训练预处理的 `[mean,std,max,min]`，现已按约3秒子段统计修正。刚连接且尚无首个ATT/MED时不应推理，等待30秒预热可以覆盖该阶段。

`powerspec.csv` 的历史生成链路仍存在可复现性风险：历史文件是高维频谱，而最终模型资产只有4维特征。实时端从 ThinkGear band power 或 Raw EEG 重建的功率特征只能做到结构兼容。正式域适配必须比较实采数据缩放后的分布，并优先用同设备数据微调。

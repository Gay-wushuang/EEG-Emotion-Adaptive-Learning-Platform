# EEG 情绪实时推理系统

## 概述

这是一个基于脑电图 (EEG) 数据的实时情绪识别系统，支持与 Unity 等应用程序通过 WebSocket 通信。

## 功能特性

- 实时 EEG 数据采集（支持 ThinkGear 设备）
- 多模态情绪分类（happy, sad, normal）
- 滑动窗口投票机制，提高识别稳定性
- WebSocket 服务器，方便与 Unity 等应用集成
- Mock 模式，使用训练数据进行测试

## 系统要求

- Python 3.8+
- CUDA 11.x（可选，用于 GPU 加速）
- ThinkGear 设备（可选，用于真实数据采集）

## 安装依赖

```bash
cd realtime_inference
pip install -r requirements.txt
```

## 快速开始

### 1. 配置系统

编辑 `config/config.yaml` 文件：

```yaml
model:
  path: "../outputs/CNN/models/best_fold4.pt"  # 模型文件路径
  type: "multimodal_cnn"
  device: "auto"  # auto, cpu, cuda
  num_classes: 3
  modalities: ["filtered", "powerspec", "att", "med"]
  time_steps: 10
  feat_dim: 4
  use_cvae: true
  cvae_latent_dim: 64
  cvae_input_dim: 160
  dropout: 0.5
  scalers_dir: null  # 设置为 null 避免 NumPy 版本兼容性问题
  skip_scaling: true  # 使用训练数据时跳过归一化

voting:
  window_size: 10  # 投票窗口大小
  vote_threshold: 0.6  # 投票阈值
  transition_duration: 1.0  # 情绪转换持续时间（秒）
  min_stability_frames: 3  # 最小稳定帧数

unity:
  host: "localhost"
  port: 8765
  max_connections: 5
  ping_interval: 30.0
  ping_timeout: 10.0

thinkgear:
  connection_mode: "mock"  # tcp 或 mock
  use_mock: true  # 使用 mock 模式
  use_training_data: true  # 使用训练数据作为 mock
  features_dir: "../features"  # 训练数据目录
  training_data_hold_samples: 30  # 每个样本保持帧数

inference:
  inference_interval: 0.1  # 推理间隔（秒）
  max_inference_latency_ms: 200
  target_cpu_usage_percent: 30

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/realtime_inference.log"
  console_output: true
```

### 2. 运行系统

```bash
cd realtime_inference
python main.py -c config/config.yaml
```

### 3. 与 Unity 集成

在 Unity 中，使用 WebSocket 客户端连接到 `ws://localhost:8765`。

服务器会发送以下格式的 JSON 数据：

```json
{
  "emotion": "normal",
  "confidence": 0.6065,
  "transition_progress": 0.0,
  "probabilities": {
    "happy": 0.1519,
    "sad": 0.2416,
    "normal": 0.6065
  },
  "timestamp": 1715133605.077
}
```

参考 `unity/EmotionReceiver.cs` 中的代码实现 Unity 客户端。

## 配置说明

### 模型配置

- `path`: 训练好的模型文件路径（相对路径）
- `device`: 推理设备，`auto` 会自动选择 GPU 或 CPU
- `scalers_dir`: 归一化器目录，设置为 `null` 可避免 NumPy 版本兼容性问题
- `skip_scaling`: 如果使用已经归一化的训练数据，设置为 `true`

### 投票配置

- `window_size`: 滑动窗口大小，越大越稳定但延迟越高
- `vote_threshold`: 情绪切换所需的最小投票比例
- `transition_duration`: 情绪平滑过渡的持续时间

### ThinkGear 配置

- `connection_mode`: 
  - `tcp`: 连接真实的 ThinkGear 设备
  - `mock`: 使用模拟数据
- `use_training_data`: 在 mock 模式下使用真实训练数据
- `training_data_hold_samples`: 每个训练样本保持的帧数，控制情绪变化速度

### Unity 配置

- `host`: WebSocket 服务器主机地址
- `port`: WebSocket 服务器端口

## 使用模式

### 1. Mock 模式（推荐用于测试）

使用训练数据作为输入，方便测试系统功能：

```yaml
thinkgear:
  connection_mode: "mock"
  use_mock: true
  use_training_data: true
```

### 2. 真实设备模式

连接 ThinkGear 设备进行实时采集：

```yaml
thinkgear:
  connection_mode: "tcp"
  com_port: "COM3"  # 根据实际情况修改
  baud_rate: 57600
  tcp_host: "127.0.0.1"
  tcp_port: 13854
  use_mock: false
```

## 输出文件

- `logs/realtime_inference.log`: 详细的系统日志
- `log.txt`: 简洁的推理结果日志

## 常见问题

### NumPy 版本兼容性问题

如果遇到 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` 错误：

1. 在配置文件中设置 `scalers_dir: null`
2. 确保 `skip_scaling: true`（如果使用训练数据）

### 模型文件路径问题

确保模型路径使用正确的相对路径：

- 从 `realtime_inference` 目录运行时，模型路径相对于该目录
- 例如：`../outputs/CNN/models/best_fold4.pt`

### Unity 连接问题

- 确保先启动推理系统，再启动 Unity 客户端
- 检查防火墙设置
- 确认端口号配置一致

## 性能优化

- 使用 GPU 加速（设置 `device: "cuda"`）
- 调整 `inference_interval` 控制推理频率
- 减小 `window_size` 降低延迟（但可能降低稳定性）

## 项目结构

```
realtime_inference/
├── main.py                 # 主程序入口
├── config/
│   └── config.yaml        # 配置文件
├── src/
│   ├── model.py           # 模型加载和推理
│   ├── voting.py          # 滑动窗口投票
│   ├── unity_comm.py      # Unity WebSocket 通信
│   └── thinkgear.py       # ThinkGear 数据采集
├── unity/
│   └── EmotionReceiver.cs # Unity 客户端示例
├── logs/                  # 日志目录
└── README.md             # 本文件
```

## 许可证

请参考项目根目录的许可证文件。

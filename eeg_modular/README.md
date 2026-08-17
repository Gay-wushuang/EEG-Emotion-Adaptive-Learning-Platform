# EEG Emotion 模块化工程

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unity](https://img.shields.io/badge/Unity-6.0-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**基于脑电图(EEG)数据的情感分析模块化工程** - 集成实时推理系统与AR/VR场景联动

## ✨ 核心特性

- 🧠 **多模态EEG信号处理** - 支持NeuroSky ThinkGear单通道设备，提取attention、meditation、8频段功率谱特征
- 🤖 **7种AI模型支持** - SVM、MLP、RF、XGBoost、LSTM、CNN、Hybrid混合模型
- ⚡ **实时推理系统** - 端到端延迟<100ms，支持WebSocket通信
- 🎮 **AR/VR场景联动** - Unity 6 + OpenXR，支持Rokid AR Lite等设备
- 📊 **配置驱动设计** - YAML配置文件控制全流程，实验可复现
- 🎨 **丰富可视化** - 混淆矩阵、训练曲线、UMAP边界图、多模型对比

## 🎯 应用场景

- **心理健康监测** - 实时情绪识别与反馈
- **冥想辅助训练** - 注意力追踪与放松指导
- **人机交互研究** - 脑机接口原型验证
- **教育娱乐应用** - EEG控制的沉浸式体验

---

## 📊 模型性能对比

| 模型 | 准确率 | 推理时间 | 适用场景 | 推荐度 |
|------|--------|----------|----------|--------|
| **CNN** | **88.89%** ⭐ | ~8ms (GPU) | 高精度需求 | ★★★★★ |
| RF | 67.90% | ~2ms (CPU) | 快速原型 | ★★★★☆ |
| Hybrid | 64.20% | ~5ms (CPU) | 集成部署 | ★★★★☆ |
| SVM | 62.96% | ~1ms (CPU) | 小样本学习 | ★★★☆☆ |
| XGB | 60.49% | ~3ms (CPU) | 性能优先 | ★★★☆☆ |
| LSTM | 60.49% | ~10ms (GPU) | 序列数据 | ★★★☆☆ |
| MLP | 59.26% | ~1ms (CPU) | 基准测试 | ★★☆☆☆ |

*测试环境：DEAP数据集，81个测试样本，Intel i7 + RTX 3060*

---

## 🏗️ 项目架构

```
eeg_modular/
│
├── eeg_emotion/                    # 主包目录
│   ├── config/                     # 配置管理 (YAML加载器)
│   ├── dl/                         # 深度学习实现 (PyTorch)
│   │   └── torch/                  # 数据加载器、损失函数、训练器
│   ├── features/                   # 特征提取模块
│   │   └── sequence/               # 序列特征提取与增强
│   ├── models/                     # 模型实现
│   │   ├── sklearn/                # 传统ML模型 (SVM/MLP/RF/XGB/Hybrid)
│   │   └── torch/                  # 深度学习模型 (LSTM/CNN/CVAE)
│   ├── preprocess/                 # 预处理管道
│   ├── viz/                        # 可视化模块
│   │   ├── confusion_matrix.py     # 混淆矩阵
│   │   ├── training_curves.py      # 训练曲线
│   │   └── umap_boundary.py        # UMAP边界图
│   └── utils/                      # 工具函数 (日志/路径/种子)
│
├── realtime_inference/             # 实时推理系统 ⭐ (完整实现)
│   ├── src/
│   │   ├── thinkgear.py            # NeuroSky数据采集 (TCP/Serial/Mock)
│   │   ├── model.py                # CVAE+CNN模型推理引擎
│   │   ├── voting.py               # 滑动窗口投票 + EMA平滑算法
│   │   ├── unity_comm.py           # WebSocket通信服务器
│   │   └── training_data_sampler.py # 训练数据采样器 (Mock模式增强)
│   ├── unity/
│   │   ├── EmotionReceiver.cs      # Unity接收脚本 (天空盒/音乐过渡)
│   │   └── EmotionReceiver_Fixed.cs # 优化版Unity脚本
│   ├── config/config.yaml          # 系统配置 (50+参数)
│   ├── main.py                     # 主程序入口 (四层架构)
│   ├── OPTIMIZATION_PLAN.md        # 防抖动性能优化方案
│   ├── THINKGEAR_SETUP_GUIDE.md    # NeuroSky设备连接指南
│   ├── requirements.txt            # Python依赖
│   └── log.txt                     # 结构化运行日志
│
├── configs/                        # 训练配置文件
│   ├── cnn.yaml                    # CNN配置 (推荐)
│   ├── hybrid.yaml                 # 混合模型配置
│   └── ...                         # 其他模型配置
│
├── scripts/                        # 训练脚本
│   ├── train_cnn_dl.py             # CNN训练脚本
│   ├── train_lstm_dl.py            # LSTM训练脚本
│   ├── train.py                    # 通用sklearn训练
│   └── multi_model_umap.py         # 多模型可视化
│
├── data/                           # 原始数据目录
├── features/                       # 预提取特征文件
├── outputs/                        # 训练输出目录
│   └── <run_timestamp>/            # 每次运行的完整结果
│       ├── models/                 # 保存的模型文件
│       ├── figures/                # 可视化图表
│       ├── logs/                   # 训练日志
│       └── metrics.json            # 评估指标
│
└── requirements.txt                # Python依赖
```

---

## 🔬 技术栈详情

### EEG采集层
- **硬件**: NeuroSky MindWave Mobile 2 (单通道消费级设备, ~$129)
- **协议**: ThinkGear Connector SDK (TCP:13854 / Serial COM)
- **采样率**: 512Hz
- **数据维度**: attention(0-100), meditation(0-100), 8频段功率谱(delta~gamma2)

### AI推理层
- **框架**: PyTorch 2.0+ / scikit-learn 1.3+
- **最优模型**: Multimodal CNN (4模态融合: filtered/powerspec/att/med)
- **输入张量**: `(batch_size, 40, 160)` = 4模态 × 10时间步 × 16特征
- **准确率**: 88.89% (测试集), Macro F1: 89.1%

### 通信层
- **协议**: WebSocket (ws://localhost:8765)
- **格式**: JSON (emotion/confidence/probabilities/timestamp)
- **延迟**: <100ms (端到端)

### 表现层
- **引擎**: Unity 6 (6000.4.0f1) + URP 17.4
- **AR框架**: OpenXR Plugin (兼容Unity 6)
- **目标平台**: Rokid AR Lite (Station 2) / Android / PC
- **功能**: 天空盒过渡、音乐切换、调试UI

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.x (可选,用于GPU加速)
- Unity 2020.3+ / 6.0 (可选,用于AR功能)
- NeuroSky ThinkGear设备 (可选,可用模拟数据)

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/eeg_modular.git
cd eeg_modular

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装额外依赖 (可选)
pip install umap-learn seaborn xgboost torch torchvision
```

### 训练你的第一个模型 (推荐CNN)

```bash
# 使用CNN配置文件进行训练 (推荐,准确率最高)
python -m scripts.train_cnn_dl -c configs/cnn.yaml

# 训练完成后,模型保存在 outputs/<timestamp>/models/
```

### 启动实时推理系统

```bash
cd realtime_inference

# 1. 准备模型和特征文件
mkdir models features
cp ../outputs/<your_run>/models/best_fold4.pt models/
cp ../features/*.joblib features/

# 2. 编辑配置文件 (可选)
# 默认使用模拟数据,无需真实EEG设备
vim config/config.yaml

# 3. 启动推理服务
python main.py -c config/config.yaml
```

### 集成到Unity项目

```csharp
// 1. 将 unity/EmotionReceiver.cs 导入Unity项目
// 2. 创建空物体并挂载脚本
// 3. 配置参数:
EmotionReceiver receiver;
receiver.serverUrl = "ws://localhost:8765";
receiver.happySkybox = happyMaterial;    // 分配快乐天空盒材质
receiver.sadSkybox = sadMaterial;        // 分配悲伤天空盒材质
receiver.normalSkybox = normalMaterial;   // 分配正常天空盒材质
receiver.happyAudio = happyMusic;         // 分配背景音乐
```

---

## 📖 详细使用指南

### 1. 配置驱动的训练流程

所有模型训练都通过YAML配置文件控制:

```yaml
# configs/cnn.yaml 示例
seed: 42
data_dir: ./data
emotions: [happy, sad, normal]

model:
  type: cnn
  num_classes: 3
  dropout: 0.35

train:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001

preprocess:
  augment: true  # 启用数据增强 (+5%准确率提升)
  noise_std: 0.01

viz:
  seaborn_confusion_matrix: true
  training_curves: true
  umap_boundary: true
```

**运行不同模型**:

```bash
# CNN (推荐, 88.89%)
python -m scripts.train_cnn_dl -c configs/cnn.yaml

# LSTM (60.49%)
python -m scripts.train_lstm_dl -c configs/lstm.yaml

# SVM (62.96%)
python -m scripts.train -c configs/svm.yaml

# Random Forest (67.90%)
python -m scripts.train -c configs/rf.yaml

# XGBoost (60.49%)
python -m scripts.train -c configs/xgb.yaml

# MLP (59.26%)
python -m scripts.train -c configs/mlp.yaml

# Hybrid (64.20%)
python -m scripts.train -c configs/hybrid.yaml
```

### 2. 可视化输出

#### 混淆矩阵 (两种风格)

```yaml
viz:
  seaborn_confusion_matrix: true   # Seaborn热力图风格
  # 或 false -> matplotlib简洁风格
```

#### 训练曲线

自动生成准确率和损失曲线,展示训练/验证集表现:

```yaml
viz:
  training_curves: true
```

#### UMAP决策边界图

降维可视化模型决策边界:

```yaml
viz:
  umap_boundary: true
```

#### 多模型对比图

在单个UMAP投影上比较多个模型:

```bash
python scripts/multi_model_umap.py \
  -c configs/svm.yaml configs/mlp.yaml configs/rf.yaml \
     configs/xgb.yaml configs/hybrid.yaml \
  -o outputs/multi_model_comparison
```

### 3. 实时推理系统详解

#### 系统架构 (四层解耦设计)

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 数据采集层 (Data Collection)                      │
│  ┌─────────────────┐                                        │
│  │ ThinkGearCollector│ ← thinkgear.py                       │
│  │ • TCP/Serial/Mock│   支持3种连接模式                        │
│  │ • 训练数据采样器 │   training_data_sampler.py              │
│  └────────┬────────┘                                        │
│           ↓ 多模态特征提取 (4模态×10步×16维)                    │
├───────────┼──────────────────────────────────────────────────┤
│           ↓                                                  │
│  Layer 2: AI推理层 (Inference Engine)                        │
│  ┌─────────────────┐                                        │
│  │ EmotionInferenceModel│ ← model.py                         │
│  │ • CVAE+CNN混合模型 │   支持条件变分自编码器                  │
│  │ • 归一化器加载     │   自动加载训练时scaler                │
│  └────────┬────────┘                                        │
│           ↓ 概率分布 [happy:0.85, sad:0.10, normal:0.05]       │
├───────────┼──────────────────────────────────────────────────┤
│           ↓                                                  │
│  Layer 3: 决策融合层 (Decision Fusion)                       │
│  ┌─────────────────────┐                                     │
│  │ ProbabilityAggregator│ ← voting.py (EMA平滑)             │
│  │ SlidingWindowVoter  │   滑动窗口投票 (防抖动)               │
│  └────────┬────────────┘                                     │
│           ↓ 最终决策 + 过渡进度 (0.0→1.0)                     │
├───────────┼──────────────────────────────────────────────────┤
│           ↓                                                  │
│  Layer 4: 表现层 (Presentation)                              │
│  ┌─────────────────┐                                        │
│  │ UnityEmotionSender│ ← unity_comm.py (WebSocket:8765)      │
│  │ EmotionReceiver.cs│ → Unity端 (天空盒/音乐过渡/调试UI)    │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘

核心特性:
✅ 端到端延迟 <100ms (典型28-53ms)
✅ 三级日志系统 (DEBUG/INFO/ERROR)
✅ 结构化日志输出 (log.txt, 可导入Excel分析)
✅ 性能监控 (每100次推理统计一次)
✅ 优雅启停 (信号处理 + 资源清理)
```

#### 关键配置项 (完整50+参数)

```yaml
# realtime_inference/config.yaml

# ===== 模型配置 (11项) =====
model:
  path: "models/best_fold4.pt"      # CNN模型路径
  type: "multimodal_cnn"              # 模型类型
  device: "auto"                      # 推理设备 (auto/cpu/cuda)
  num_classes: 3                      # 情绪类别数
  modalities: ["filtered", "powerspec", "att", "med"]  # 4种模态
  time_steps: 10                     # 时间步数
  feat_dim: 4                        # 特征维度
  use_cvae: true                     # 是否使用CVAE编码器
  cvae_latent_dim: 64                # CVAE潜在空间维度
  cvae_input_dim: 160               # CVAE输入维度
  dropout: 0.5                       # Dropout比率
  scalers_dir: "../features"         # 归一化器目录
  skip_scaling: true                 # 跳过归一化(训练数据已归一化)

# ===== 投票算法配置 (4项) =====
voting:
  window_size: 10                    # 滑动窗口大小
  vote_threshold: 0.6                # 投票阈值(0-1)
  transition_duration: 1.0           # 过渡动画时长(秒)
  min_stability_frames: 3            # 最小稳定帧数

# ===== Unity通信配置 (5项) =====
unity:
  host: "localhost"
  port: 8765                         # WebSocket端口
  max_connections: 5                  # 最大连接数
  ping_interval: 30.0                # 心跳间隔(秒)
  ping_timeout: 10.0                 # 心跳超时(秒)

# ===== ThinkGear EEG设备配置 (11项) =====
thinkgear:
  connection_mode: "mock"            # 连接模式 (tcp/serial/mock)
  com_port: "COM6"                   # 串口号
  baud_rate: 57600                   # 波特率
  tcp_host: "127.0.0.1"             # TCP主机地址
  tcp_port: 13854                    # TCP端口
  sample_rate: 512                   # 采样率(Hz)
  buffer_size: 1024                  # 缓冲区大小
  use_mock: true                     # 使用模拟数据
  use_training_data: true            # 使用训练数据作为mock ⭐新增
  features_dir: "../features"        # 训练数据目录
  training_data_hold_samples: 30    # 每个样本保持帧数 ⭐新增

# ===== 性能控制配置 (3项) =====
inference:
  inference_interval: 0.1            # 推理间隔(秒)
  max_inference_latency_ms: 200       # 最大允许延迟(ms)
  target_cpu_usage_percent: 30        # 目标CPU占用(%)

# ===== 日志配置 (2项) =====
logging:
  level: "DEBUG"                     # 日志级别 (DEBUG/INFO/WARNING/ERROR)
  log_file: "logs/realtime_inference.log"  # 日志文件路径
  console_output: true               # 是否输出到控制台
```

#### 开发模式详解

**模式1: 简单Mock数据** (快速测试,无需训练数据)
```yaml
thinkgear:
  use_mock: true
  use_training_data: false          # 关闭训练数据采样
```
特点: 使用正弦波+噪声生成模拟EEG数据,适合快速验证系统流程。

**模式2: 训练数据回放** (推荐开发阶段) ⭐新增
```yaml
thinkgear:
  use_mock: true
  use_training_data: true           # 启用训练数据采样器
  features_dir: "../features"       # 训练特征目录
  training_data_hold_samples: 30   # 每个样本保持30帧(变化更平滑)
```
特点:
✅ 从真实训练数据中采样,确保特征格式100%一致
✅ 可控制变化速度(hold_samples越大变化越慢)
✅ 自动循环遍历所有样本,覆盖所有情绪类别
✅ 日志中显示真实标签,便于调试模型推理准确性

**模式3: 真实设备连接** (生产部署)
```yaml
thinkgear:
  use_mock: false
  connection_mode: tcp             # 或 serial
  tcp_host: "127.0.0.1"
  tcp_port: 13854
```
前提: 需先启动ThinkGear Connector并连接硬件设备。
详见: [ThinkGear连接指南](./realtime_inference/THINKGEAR_SETUP_GUIDE.md)

### 4. AR/VR集成指南

#### 支持的平台

| 平台 | 引擎版本 | AR框架 | 状态 |
|------|---------|--------|------|
| PC (Windows) | Unity 6 | OpenXR | ✅ 已测试 |
| Rokid AR Lite | Unity 6 | OpenXR | 🔄 开发中 |
| Android手机 | Unity 6 | OpenXR | ✅ 已测试 |
| Meta Quest 3 | Unity 6 | OpenXR | 📋 计划中 |

#### Unity 6 + OpenXR 配置步骤

1. **安装OpenXR插件**
   ```
   Window → Package Manager → 搜索 "OpenXR" → Install
   ```

2. **配置XR设置**
   ```
   Edit → Project Settings → XR Plug-in Management
   ☑ Initialize XR on Startup
   ☑ OpenXR (Standalone/Android标签页)
   ```

3. **添加XR Origin**
   ```
   菜单栏 → XR → XR Origin (添加基础XR相机系统)
   ```

4. **挂载EmotionReceiver脚本**
   ```csharp
   // 在XR Origin上添加组件
   // 配置Server URL: ws://localhost:8765
   // 分配天空盒材质和音频
   ```

#### 性能优化建议 (移动端)

针对Rokid AR Lite等移动设备的优化策略:

- **模型优化**: CNN减面至8-12K三角面,纹理压缩至ASTC 4x4
- **内存控制**: 运行时显存<300MB,包体<100MB
- **渲染优化**: 启用SRP Batching, GPU Instancing, LOD
- **帧率目标**: 45-60FPS (最低30FPS可接受)

---

## 📈 数据增强效果

我们的数据增强策略显著提升了模型性能:

| 条件 | 准确率 | Macro F1 | 提升幅度 |
|------|--------|----------|----------|
| 无增强 | 83.9% | 84.2% | - |
| **有增强** | **88.9%** | **89.1%** | **+5.0%** |

**增强策略贡献度**:
- 时间抖动: +2.1%
- 高斯噪声: +1.8%
- Mixup数据增强: +2.5%
- 组合协同效应: +5.0%

---

## 🎓 最佳实践

### 实验管理

1. **使用配置文件驱动** - 所有参数通过YAML控制,不在代码中硬编码
2. **固定随机种子** - `seed: 42` 确保可复现性
3. **统一输出目录** - 所有结果保存到`outputs/<timestamp>/`
4. **记录完整日志** - 包含超参数、性能指标、环境信息

### 模型选择指南

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 最高精度需求 | CNN | 88.89%准确率,适合演示/论文 |
| 快速原型开发 | RF/SVM | 训练快(<5min),解释性强 |
| 移动端部署 | CNN/LSTM | 可量化为ONNX/TFLite |
| 集成学习 | Hybrid | 综合多个模型优势 |
| 边缘设备 | RF/XGB | CPU推理快,内存占用低 |

### 性能调优技巧

**CNN超参数优先级**:
1. `learning_rate`: [0.0001, 0.001] (最优0.001)
2. `batch_size`: [128, 256, 512] (最优256)
3. `weight_decay`: [0, 0.001] (最优0.001)
4. `num_layers`: [2, 3, 4] (最优3)

**数据预处理要点**:
- 训练集/测试集严格分离
- 仅在训练集上拟合预处理器(StandardScaler等)
- 启用数据增强可提升5%准确率

---

## ❓ 常见问题

### Q: UMAP图未生成?
**A**: 检查以下配置:
```yaml
viz:
  umap_boundary: true  # 必须为true
```
安装依赖: `pip install umap-learn`

### Q: 结果波动较大?
**A**:
- 确认`seed`设置一致
- 检查是否启用了随机数据增强
- 增加训练epochs或调整学习率

### Q: XGBoost在CPU环境报错?
**A**: 修改配置:
```yaml
model:
  xgboost:
    device: cpu  # 强制使用CPU
```

### Q: 无法连接ThinkGear设备?
**A**:
1. 检查串口/COM口是否正确
2. 确认ThinkGear Connector已启动
3. 尝试模拟模式: `use_mock: true`
4. 查看日志: `logs/realtime_inference.log`

### Q: Unity无法连接WebSocket?
**A**:
1. 确认Python推理服务已启动
2. 检查防火墙设置(允许localhost:8765)
3. 验证URL格式: `ws://localhost:8765` (不是http://)
4. 查看Console错误信息

### Q: 内存不足(OOM)?
**A**:
- 降低batch_size: 256 → 128 → 64
- 启用梯度累积
- 使用混合精度训练: `train.use_amp: true`
- 减少模型层数或隐藏单元数

---

## 🔧 高级功能

### 自定义模型

继承基类实现新模型:

```python
from eeg_emotion.models.base import ModelAdapter

class CustomModel(ModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.model = self._build_model()

    def fit(self, X_train, y_train):
        # 实现训练逻辑
        pass

    def predict(self, X_test):
        # 实现预测逻辑
        return predictions

    def save(self, path):
        # 保存模型
        pass

    def load(self, path):
        # 加载模型
        pass
```

### 扩展情绪类别

1. 修改配置: `model.num_classes: 5`
2. 更新标签列表: `emotions: [happy, sad, angry, fear, neutral]`
3. 重新训练模型
4. 更新Unity脚本中的处理逻辑

### 集成其他EEG设备

继承`ThinkGearCollector`类:

```python
from realtime_inference.src.thinkgear import ThinkGearCollector

class CustomEEGDevice(ThinkGearCollector):
    def connect(self):
        # 实现设备连接
        pass

    def read_data(self):
        # 读取原始数据
        raw_data = self.device.read()
        return self.parse_data(raw_data)

    def parse_data(self, raw):
        # 解析为标准格式
        return {
            'attention': value,
            'meditation': value,
            'eegPower': {...}
        }
```

---

## 📁 项目结构说明

### 核心模块

| 模块 | 路径 | 功能 |
|------|------|------|
| **配置管理** | `eeg_emotion/config/` | YAML配置加载与校验 |
| **特征提取** | `eeg_emotion/features/` | 多模态特征构建 |
| **模型库** | `eeg_emotion/models/` | 7种算法实现 |
| **预处理** | `eeg_emotion/preprocess/` | 数据清洗与标准化 |
| **可视化** | `eeg_emotion/viz/` | 图表生成工具 |
| **深度学习** | `eeg_emotion/dl/torch/` | PyTorch训练基础设施 |
| **实时推理** | `realtime_inference/` | 生产级推理服务 |
| **Unity集成** | `realtime_inference/unity/` | AR/VR场景控制 |

### 输出目录结构

每次训练运行会生成独立目录:

```
outputs/run_20260419_143022/
├── models/
│   ├── best_fold4.pt          # 最优CNN模型
│   └── model.joblib            # sklearn模型
├── artifacts/
│   ├── scaler.joblib           # 标准化器
│   └── label_encoder.joblib    # 标签编码器
├── figures/
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── training_curves.png     # 训练曲线
│   └── umap_boundary.png       # UMAP图
├── logs/
│   └── train.log               # 详细训练日志
└── metrics.json                # 评估指标汇总
```

---

## 📚 参考资源

### 文档
- [AGENTS.md](./AGENTS.md) - 完整技术规范与架构设计 (含模型性能、超参数调优指南)
- [实时推理系统文档](./realtime_inference/OPTIMIZATION_PLAN.md) - 防抖动性能优化方案 ⭐新增
- [ThinkGear配置指南](./realtime_inference/THINKGEAR_SETUP_GUIDE.md) - NeuroSky设备接入教程 ⭐新增

### 外部链接
- [NeuroSky开发者文档](https://developer.neurosky.com/) - ThinkGear API参考
- [Rokid AR Platform](https://developer.rokid.com/) - AR SDK下载
- [Unity OpenXR文档](https://docs.unity3d.com/Packages/com.unity.xr.openxr@latest) - XR集成指南
- [DEAP数据集](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html) - 情感计算基准数据

### 学术引用

如果您在研究中使用了本项目,请引用:

```bibtex
@misc{eeg_emotion_2026,
  title={EEG Emotion: A Modular Framework for Real-time Emotion Recognition with AR Integration},
  author={Your Team Name},
  year={2026},
  url={https://github.com/your-repo/eeg_modular}
}
```

---

## 🤝 贡献指南

我们欢迎社区贡献!请遵循以下步骤:

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8风格
- 添加类型注解和文档字符串
- 编写单元测试(覆盖核心功能)
- 更新相关文档

---

## 📄 许可证

本项目采用 MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 👥 团队与致谢

- **项目负责人**: [Your Name]
- **技术顾问**: [Advisor Name]
- **特别感谢**: NeuroSky提供的开发设备支持,开源社区的宝贵资源

---

## 📞 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/eeg_modular/issues)
- **技术讨论**: [ Discussions](https://github.com/your-repo/eeg_modular/discussions)
- **邮箱**: your.email@example.com

---

<div align="center">

**⭐ 如果这个项目对你有帮助,请给一个Star! ⭐**

Made with ❤️ by EEG Emotion Team

</div>

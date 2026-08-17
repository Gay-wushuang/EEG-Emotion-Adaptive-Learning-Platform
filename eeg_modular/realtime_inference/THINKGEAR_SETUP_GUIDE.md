# NeuroSky ThinkGear 设备连接指南

本指南将帮助你连接真实的 NeuroSky ThinkGear 设备进行脑电数据采集。

---

## 📋 前置要求

### 1. 硬件准备
- NeuroSky 脑电设备（MindWave, MindSet 等）
- USB 蓝牙适配器（如使用无线设备）
- 设备已佩戴好，接触良好

### 2. 软件准备
- ThinkGear Connector（已包含在项目中）
- Python 依赖已安装（`pip install -r requirements.txt`）

---

## 🚀 连接步骤（推荐 TCP 模式）

### 第一步：启动 ThinkGear Connector

1. 打开文件夹：`d:\proegg\eeg_modular\ThinkGear Connector\`
2. 双击运行：`ThinkGear Connector.exe`
3. 等待程序启动，任务栏会出现图标

### 第二步：连接设备

在 ThinkGear Connector 中：

1. **串口模式**（推荐）
   - 点击菜单 → 连接 → 串口
   - 选择你的设备对应的串口号（如 COM6）
   - 点击"连接"

2. **蓝牙模式**
   - 确保设备已配对
   - 点击菜单 → 连接 → 蓝牙
   - 选择你的设备
   - 点击"连接"

### 第三步：确认数据接收

- 观察 ThinkGear Connector 界面
- 应该能看到实时的脑电信号波形
- Attention 和 Meditation 数值应该在 0-100 之间变化
- 确保设备指示灯正常闪烁

---

## 🔧 配置系统（TCP 模式）

### 1. 确认配置文件

编辑 `config/config.yaml`，确保以下设置：

```yaml
thinkgear:
  connection_mode: "tcp"      # ← 确保是 "tcp"
  com_port: "COM6"             # ← 修改为你的实际串口号
  baud_rate: 57600
  tcp_host: "127.0.0.1"        # ← ThinkGear Connector地址
  tcp_port: 13854               # ← ThinkGear Connector默认端口
  sample_rate: 512
  buffer_size: 1024
  use_mock: false              # ← 确保是 false
```

### 2. 验证配置

检查以下参数：
- `connection_mode` 必须是 `"tcp"`
- `use_mock` 必须是 `false`
- `tcp_host` 和 `tcp_port` 通常不需要修改（127.0.0.1:13854）

---

## ▶️ 启动系统

### 1. 先停止当前程序

如果之前的程序还在运行，按 `Ctrl+C` 停止

### 2. 重新启动推理系统

```bash
cd realtime_inference
python main.py -c config/config.yaml
```

### 3. 观察启动日志

你应该看到：
```
ThinkGearCollector initialized for TCP: 127.0.0.1:13854
ThinkGearCollector started
Connecting to ThinkGear Connector at 127.0.0.1:13854...
Connected to ThinkGear Connector!
```

---

## 📊 数据采集说明

### ThinkGear Connector JSON 数据格式

系统从 ThinkGear Connector 读取以下 JSON 数据：

```json
{
  "rawEeg": -2048,
  "eSense": {
    "attention": 50,
    "meditation": 50
  },
  "eegPower": {
    "delta": 100000,
    "theta": 50000,
    "lowAlpha": 20000,
    "highAlpha": 20000,
    "lowBeta": 10000,
    "highBeta": 10000,
    "lowGamma": 5000,
    "highGamma": 5000
  }
}
```

### 数据说明

| 数据类型 | 说明 | 用途 |
|---------|------|------|
| **rawEeg** | 原始脑电信号 | filtered 模态 |
| **eSense/attention** | 专注度值 | att 模态 |
| **eSense/meditation** | 冥想度值 | med 模态 |
| **eegPower** | 各频段功率 | powerspec 模态 |

### 频段说明

| 频段 | 频率范围 | 关联状态 |
|------|---------|---------|
| Delta | 0.5-4Hz | 深度睡眠 |
| Theta | 4-8Hz | 困倦、冥想 |
| Alpha | 8-13Hz | 放松、清醒 |
| Beta | 13-30Hz | 专注、警觉 |
| Gamma | 30-100Hz | 高级认知 |

---

## 🔄 其他连接模式

### 串口模式（Serial）

如需直接连接串口（不经过 ThinkGear Connector）：

```yaml
thinkgear:
  connection_mode: "serial"   # ← 修改为 "serial"
  com_port: "COM6"
  baud_rate: 57600
  use_mock: false
```

### 模拟模式（Mock）

用于开发测试，无需真实设备：

```yaml
thinkgear:
  connection_mode: "mock"     # ← 或 use_mock: true
  use_mock: true
```

---

## ⚠️ 常见问题排查

### 问题1：无法连接到 ThinkGear Connector

**错误信息**：`Could not connect to ThinkGear Connector at 127.0.0.1:13854`

**解决方案**：
1. 确认 ThinkGear Connector 正在运行
2. 检查 ThinkGear Connector 是否启用了 TCP 输出
3. 查看 ThinkGear Connector 设置中的端口号是否为 13854
4. 检查防火墙设置

### 问题2：ThinkGear Connector 显示"未与任何应用软件连接"

**现象**：ThinkGear Connector 界面显示此消息

**解决方案**：
- ✅ **这是正常的！** - 只有当有客户端（我们的系统）连接时，这个消息才会消失
- 继续启动我们的系统，连接成功后消息会消失

### 问题3：连接成功但没有数据

**现象**：系统显示已连接，但推理没有变化

**解决方案**：
1. 确认设备已正确佩戴
2. 检查 ThinkGear Connector 界面是否有信号
3. 查看日志文件中的详细数据
4. 尝试重新连接设备

### 问题4：数据质量差

**现象**：信号不稳定，数值跳动大

**解决方案**：
- 确保设备佩戴正确，接触良好
- 检查周围是否有电磁干扰
- 保持安静的环境
- 重新佩戴设备

---

## 📝 使用建议

### 1. 佩戴建议
- 保持设备清洁
- 确保传感器接触皮肤
- 避免剧烈运动
- 保持舒适的姿势

### 2. 环境建议
- 减少电磁干扰
- 保持安静环境
- 避免强光直射
- 适宜的温度和湿度

### 3. 数据采集建议
- 每次采集前让用户静坐1-2分钟
- 记录用户状态和环境信息
- 定期检查设备连接状态
- 注意电池电量（如使用无线设备）

### 4. ThinkGear Connector 使用建议
- 保持 ThinkGear Connector 始终运行
- 不要同时运行多个连接到 ThinkGear Connector 的程序
- 定期检查 ThinkGear Connector 的日志

---

## 🔄 切换连接模式

### 从 TCP 切换到串口

修改配置：
```yaml
thinkgear:
  connection_mode: "serial"
  com_port: "COM6"
```

### 从任意模式切换到模拟

修改配置：
```yaml
thinkgear:
  connection_mode: "mock"
  use_mock: true
```

---

## 📞 获取帮助

如遇到问题：
1. 查看日志文件：`logs/realtime_inference.log`
2. 检查 ThinkGear Connector 的日志
3. 确认设备工作正常

---

**祝使用愉快！** 🧠✨

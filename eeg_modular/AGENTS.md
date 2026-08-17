# AGENTS.md — 智学脑机助手协作与交付规范

本文件是仓库内所有 AI Agent、IDE Agent 和人工开发者的最高级项目协作说明。开始工作前必须完整阅读。若任务目录中存在更深层的 `AGENTS.md`，则该文件仅覆盖其所在子目录。

## 1. 当前目标

在比赛截止前交付一个可安装、可演示、可回放、可审计的 Windows 桌面应用：

> 面向智慧教育的脑机接口学习状态辅助系统，通过 MindWave 单通道 EEG、三分类状态概率、信号质量门控、长期趋势平滑和 AI 学习反馈，形成采集—分析—反馈—报告闭环。

当前阶段是**软件产品化与成果交付**。除非用户明确要求，不再扩大模型搜索、重新定义标签或重做 UI 技术栈。

## 2. 不可夸大的科学边界

### 标签契约

模型内部索引保持不变：

```text
0 happy  → positive（积极）
1 normal → neutral（中性）
2 sad    → negative（负性）
```

`sad` 是历史兼容字段。对外可显示 `negative`，但不得修改模型输出顺序。负性状态不等同于疲劳、任务负荷、眨眼、肌电、动作伪迹或信号变差。

### 当前允许引用的指标

```text
原 Production Baseline v1：
严格受试者隔离三分类全覆盖 Accuracy 63.88%
Macro-F1 62.69%

v3 最佳通用候选：
CNN + 非线性树模型等权集成 Accuracy 66.41%
Macro-F1 66.30%

3-shot 个人原型初始化：
三分类全覆盖 Accuracy 72.01%
Macro-F1 68.11%

高置信度选择性识别：
接受窗口 Accuracy 90.20%
Coverage 18.41%
Macro-F1 82.86%
```

90.20% 不是全覆盖准确率。引用时必须在同一句或同一张图中标注“高置信度接受窗口”和“覆盖率18.41%”。

旧88.89%来自已审计的历史管线，其中频谱特征实际误用0 Hz列，且评估口径不满足当前严格受试者隔离要求。它只能出现在 `legacy_model/` 或审计报告中，不得作为产品性能。

### 产品措辞

允许：

- 学习状态辅助；
- 状态趋势；
- 风险提示；
- 个性化学习建议；
- 高置信度窗口；
- 信号可信度不足，暂不解释。

禁止：

- 医疗诊断、抑郁筛查或临床判断；
- 精确读取情绪；
- 90%模型全覆盖准确率；
- 将重叠窗口视为独立样本；
- 将 `negative` 自动解释为压力、悲伤或受挫已经被正确识别。

## 3. 冻结的核心契约

除非用户明确授权，以下内容禁止修改：

- `production_baseline_v1/` 中的模型、Scaler、类别映射和哈希；
- `eeg_emotion/features/canonical.py` 的正式特征生成逻辑；
- 512 Hz Raw EEG 时间轴重建规则；
- 训练/部署使用同一特征提取器的原则；
- 内部类别顺序 `happy / normal / sad`；
- 受试者隔离评估原则；
- 测试集不得用于拟合Scaler、选择模型或产生训练伪标签；
- ATT/MED不进入 Production Baseline v1 的情绪分类张量。

ATT/MED可用于仪表盘、趋势解释、低专注提醒和反馈规则。

## 4. 三方工具分工

### Codex — 技术负责人、集成与验收

负责：

- 产品架构和跨模块数据契约；
- MindWave/ThinkGear采集；
- 信号时间轴、质量门控、特征提取和推理；
- 状态平滑、反馈逻辑、会话记录和报告导出；
- 将UI原型迁入最终应用；
- 单元测试、回放测试、打包和干净环境验收；
- 审查所有来自其他Agent的代码。

主要维护：

```text
smart_learning_app/
eeg_emotion/
realtime_inference/
tests/
production_baseline_v1/
```

### TRAE Work CN — UI视觉原型

只负责：

- PySide6页面、组件和布局；
- QSS主题、图标和视觉资源；
- `DashboardState`驱动的模拟数据展示；
- 1920×1080及1366×768适配；
- UI交互状态和空状态；
- 历史CSV回放界面的纯前端部分。

默认只允许修改：

```text
ui_prototype/
```

禁止修改：

```text
eeg_emotion/
realtime_inference/
production_baseline_v1/
legacy_model/
scripts/
splits_*/
features_*/
outputs_*/
```

TRAE Work不得自行更换为Electron、Web前端、Tkinter、Unity、Godot或C#。当前前端技术栈固定为 PySide6 + QSS + pyqtgraph。

### Trae CN — 人工预览和局部调整

负责：

- 运行UI原型；
- 调整文字、颜色、边距和响应式布局；
- 修复局部PySide6错误；
- 提交清晰的小范围改动。

Trae CN不另建第二套架构，不重写核心服务，不修改模型契约。若需要跨目录修改，先交由Codex评估。

## 5. 目录所有权

```text
eeg_modular/
├─ ui_prototype/               # TRAE Work CN独占原型区
├─ smart_learning_app/         # Codex最终集成产品
├─ eeg_emotion/                # 冻结算法核心，Codex维护
├─ realtime_inference/         # 真实采集与推理，Codex维护
├─ production_baseline_v1/     # 冻结生产资产
├─ legacy_model/               # 历史复现，不接入产品
├─ tests/                      # 自动测试
├─ data/                       # 产品运行数据；不得提交隐私数据
├─ reports/                    # 产品导出报告
├─ scripts/                    # 实验与维护脚本，不进入安装包
├─ outputs_v*/                 # 实验结果，不进入安装包
└─ AGENTS.md
```

若目录尚不存在，可按以上结构创建。任何Agent不得因为目录不存在而把原型散落到仓库根目录。

## 6. UI产品规范

### 页面范围

必须包含：

1. 欢迎与设备检查；
2. 用户与初始化；
3. 实时学习仪表盘；
4. 学习任务和事件标记；
5. 历史会话与报告；
6. 设置与系统诊断；
7. 历史CSV回放模式。

### 视觉方向

- 深蓝灰背景，青蓝主色；
- 绿色表示可信，琥珀色表示警告，红色只表示严重错误；
- 科技感但不做医疗监护仪风格；
- 禁止大面积霓虹、无意义渐变和持续动画；
- 重要状态必须同时使用文字和颜色，不能只靠颜色；
- UI必须适应1920×1080和1366×768；
- 中文文本不得截断或出现编码错误。

### 双层状态输出

界面必须将以下两者分开：

```text
模型输出：P(positive) / P(neutral) / P(negative)
质量输出：可信 / 警告 / 低可信或OOD
```

质量不合格时，保存概率日志，但界面显示：

> 当前信号质量不足，暂不进行学习状态解释。

### DashboardState接口

UI只能通过统一状态对象接收业务数据，不直接读取socket、模型或CSV。最低字段：

```python
DashboardState(
    run_id: str,
    mode: str,                       # live | replay
    connector_status: str,           # offline | connecting | online
    device_status: str,              # offline | waiting_raw | online
    sample_rate_hz: float | None,
    poor_signal: int | None,
    quality_level: str,              # trusted | warning | rejected
    quality_reasons: list[str],
    warmup_progress: float,           # 0.0 ~ 1.0
    prob_positive: float | None,
    prob_neutral: float | None,
    prob_negative: float | None,
    predicted_state: str | None,
    confidence: float | None,
    stable_state: str | None,
    attention: float | None,
    meditation: float | None,
    feedback_text: str,
    session_seconds: float,
)
```

新增字段必须保持向后兼容或同步更新mock数据、UI和测试。

## 7. 运行状态机

正式应用必须遵循：

```text
未连接
→ 正在连接Connector
→ 等待首个Raw
→ 检查信号质量
→ 填充观察缓冲
→ 正常分析
→ 信号警告/状态不确定
→ 正在结束与保存
→ 会话完成
```

要求：

- TCP连接成功不等于设备已就绪；
- 首个Raw到达后才开始预热；
- 推理观察窗口与2秒UI更新步长分开；
- 决策层汇总最近60～90秒概率；
- 重叠窗口不是独立证据；
- 信号拒识窗口不计入任何情绪；
- 设备断线、重连或严重质量异常后不得使用陈旧缓冲；
- 会话结束必须等待文件保存确认。

## 8. 线程和错误处理

- UI线程禁止执行socket读取、滤波、模型推理、CSV大文件读取或报告生成；
- 使用 `QThread`/worker和Qt signal传递不可变状态快照；
- 单条JSON解析失败必须记录并继续，不得终止整场采集；
- UI不得展示Python traceback；用户看到的是中文错误、原因和恢复操作；
- 所有未捕获异常写入带时间戳的本地日志；
- 回放模式和真实模式必须经过相同的质量、特征、推理和决策接口。

## 9. 数据与隐私

- 原始EEG默认只在本地处理；
- 不添加未经用户授权的上传、遥测或云端依赖；
- 日志不得记录姓名、邮箱或其他无关个人信息；
- 会话使用匿名 `run_id`；
- 用户可以删除历史会话；
- 演示数据和真实数据必须明确标识；
- 不提交真实受试者原始数据到Git。

## 10. Git和修改规则

建议分支：

```text
main                  # 可运行稳定版本
ui/trae-work          # TRAE Work前端原型
integration/codex     # Codex集成
```

协作规则：

- 一个文件同一时间只能由一个角色维护；
- 不覆盖或回滚不属于自己的修改；
- 不使用 `git reset --hard`、强制checkout或批量删除；
- 每次提交只解决一个清晰问题；
- 提交信息建议：`ui:`、`core:`、`fix:`、`test:`、`build:`、`docs:`；
- UI交接必须附带运行命令、截图、已知问题和修改文件列表；
- 未经Codex审查的原型代码不得进入最终安装包。

## 11. 开发与验收

### 开发入口

当前Python环境优先使用：

```powershell
E:\anaconda3\envs\eegcnn\python.exe
```

Agent不得擅自升级PyTorch、NumPy、SciPy或scikit-learn。新增UI依赖前先检查是否已安装，并把最终版本写入锁定文件。

### 必须通过的测试

提交核心改动前至少执行相关测试：

```powershell
python -m unittest discover -s tests
```

最终验收包括：

1. 无设备时能启动并解释连接问题；
2. mock ThinkGear服务器可跑通采集；
3. 历史CSV可完整回放；
4. 模型和Scaler自检通过；
5. 30秒预热及2秒更新逻辑正确；
6. 信号差时停止状态解释；
7. 会话结束后数据正常保存；
8. 中文路径下可运行；
9. 无网络时可运行；
10. 干净Windows虚拟机中无需安装Python即可启动。

## 12. 打包规则

- 使用PyInstaller `onedir`，不优先使用`onefile`；
- 最终通过Inno Setup制作安装程序；
- 安装包只包含运行所需代码、模型、配置和资源；
- 排除训练数据、实验输出、split文件、旧模型和研究脚本；
- 启动时检查ThinkGear Connector、模型资产、配置哈希、磁盘空间和写入权限；
- 提供桌面快捷方式、版本号、卸载入口和离线回放数据；
- 使用VMware干净Windows环境完成最终安装验收。

## 13. 两天冲刺优先级

按以下顺序工作，不得先做低优先级装饰：

1. 可运行主窗口和统一状态接口；
2. 真实设备连接、预热和实时推理；
3. 质量门控和状态不确定；
4. 60～90秒稳定趋势与反馈；
5. 会话保存、历史回放和事件标记；
6. 报告导出；
7. 安装包与干净环境验收；
8. 图标、动画和次要视觉润色。

最小完成标准：

> 应用可一键启动，检测Connector并连接MindWave；首个Raw后完成预热；持续显示三分类概率、信号可信度和稳定趋势；生成一次合理学习建议；能够结束会话、保存记录、回放历史数据并在无Python的干净Windows中运行。

## 14. Agent交付格式

每个Agent完成任务后必须报告：

```text
完成内容：
修改文件：
运行方法：
验证结果：
未完成/已知问题：
是否修改冻结契约：否（默认必须为否）
```

若无法证明功能已运行，不得写“已完成”；应写“已实现，尚未在目标环境验证”。

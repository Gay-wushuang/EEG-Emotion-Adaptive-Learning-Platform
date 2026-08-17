# 防抖动优化方案

## 1. 防抖动实现分析

### 两处防抖动实现：

#### 第一处（Python端）
1. **ProbabilityAggregator** - 概率平滑器
   - 使用 EMA（指数移动平均）平滑概率
   - 配置：window_size=10, alpha=0.3
   - 作用：减少单帧概率波动

2. **SlidingWindowVoter** - 滑动窗口投票器
   - 配置：window_size=10, vote_threshold=0.6, transition_duration=1.0s, min_stability_frames=3
   - 作用：基于历史投票结果决定最终情绪

#### 第二处（Unity端）
1. **天空盒平滑过渡** - transitionSmoothTime = 1秒
2. **音乐交叉淡入淡出** - musicTransitionTime = 2秒

### 当前问题：
- 防抖动过于保守，响应迟缓
- 需要10帧历史数据 + 60%投票率 + 1秒过渡时间
- 用户反馈：整体处理效果未达预期

---

## 2. 已实施的修改

### 临时禁用滑动窗口投票器
- **文件**：`main.py` 第 249-260 行
- **修改**：注释掉 `SlidingWindowVoter.update()` 调用
- **替代方案**：直接使用概率平滑后的 argmax 结果
- **保留**：ProbabilityAggregator 继续工作，作为折中方案

---

## 3. 替代优化方案

### 方案 A：调整防抖动参数（推荐）

#### 配置调整（config.yaml）
```yaml
voting:
  window_size: 5              # 从10减少到5
  vote_threshold: 0.5         # 从0.6降低到0.5
  transition_duration: 0.5    # 从1.0s减少到0.5s
  min_stability_frames: 1     # 从3减少到1
```

**预期效果**：
- 响应速度提升约 50-70%
- 仍保持一定防抖动能力
- 平衡了响应性和稳定性

**潜在风险**：
- 可能出现少量误跳
- 建议配合方案 B 使用

---

### 方案 B：节流(Throttling) + 动态阈值

#### 实现思路：
1. 保留投票器，但添加**高置信度快速通道**
2. 当某类概率 > 0.8 时，立即切换，无需等待
3. 中等置信度（0.5-0.8）时，使用投票器
4. 低置信度（< 0.5）时，保持当前状态

#### 伪代码：
```python
def update_with_throttling(self, emotion, probs, current_time):
    max_prob = probs.max()
    
    # 高置信度快速通道
    if max_prob > 0.8:
        return self._immediate_switch(emotion)
    
    # 中等置信度使用投票
    elif max_prob > 0.5:
        return self._normal_voting(emotion, probs, current_time)
    
    # 低置信度保持
    else:
        return self.current_emotion, 0.0
```

**预期效果**：
- 高置信度时响应极快（<100ms）
- 低置信度时保持稳定
- 最佳平衡方案

---

### 方案 C：动态防抖动阈值机制

#### 实现思路：
根据以下因素动态调整阈值：
1. **置信度高低**：置信度越高，阈值越低
2. **情绪变化频率**：频繁变化时降低阈值
3. **用户活动水平**（如可获取）：活动高时降低阈值

#### 动态阈值公式：
```python
def get_dynamic_threshold(self, max_prob, change_rate):
    base_threshold = 0.6
    conf_factor = (1 - max_prob) * 0.3  # 置信度越高，阈值越低
    change_factor = min(change_rate * 0.2, 0.2)  # 变化越快，阈值越低
    
    return max(0.3, base_threshold - conf_factor - change_factor)
```

**预期效果**：
- 自适应各种场景
- 无需手动调参
- 长期使用效果最佳

---

## 4. 测试对比建议

### 测试指标：
1. **响应延迟**：情绪变化到显示变化的时间
2. **误跳次数**：1分钟内不必要的情绪切换次数
3. **用户满意度**：主观评分（1-5分）

### 对比方案：
| 方案 | 响应延迟 | 误跳次数 | 推荐场景 |
|------|---------|---------|---------|
| 当前配置（已禁用投票器） | 快 | 可能较多 | 快速测试 |
| 方案 A（参数调整） | 中 | 较少 | 平衡使用 |
| 方案 B（节流+快速通道） | 快 | 少 | ⭐ 推荐 |
| 方案 C（动态阈值） | 自适应 | 最少 | 长期最优 |

---

## 5. 恢复原代码

如需恢复滑动窗口投票器，只需取消 `main.py` 第 249-260 行的注释即可。

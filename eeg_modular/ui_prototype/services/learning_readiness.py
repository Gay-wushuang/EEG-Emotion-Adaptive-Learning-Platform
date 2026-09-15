"""UI/controller guard for starting a formal student learning record."""

from services.dashboard_state import MAX_POOR_SIGNAL


def learning_start_block_reason(state, *, require_baseline: bool = True) -> str:
    """Return a user-facing reason, or an empty string when start is allowed.

    Teaching demo is intentionally independent of physical device readiness.
    Live checks reuse the existing DashboardState fields and do not introduce a
    second acquisition state machine.
    """
    if getattr(state, "mode", "live") == "mock":
        return ""
    if getattr(state, "connector_status", "offline") != "online":
        return "请先连接 EEG 设备后开始学习。"
    device = getattr(state, "device_status", "offline")
    if device == "waiting_raw":
        return "正在等待设备数据，请稍候。"
    if device != "online":
        return "请先连接 EEG 设备后开始学习。"
    poor = getattr(state, "poor_signal", None)
    if poor is None or poor >= MAX_POOR_SIGNAL or getattr(state, "quality_level", "rejected") == "rejected":
        return "当前信号质量不足，请调整设备后再开始学习。"
    if not bool(getattr(state, "warmup_complete", False)):
        return "智能分析正在准备中，请等待预热完成。"
    if str(getattr(state, "model_status", "READY")).upper() != "READY":
        return "智能分析暂不可用，请联系管理员检查系统配置。"
    if require_baseline and getattr(state, "baseline_status", "IDLE") != "COMPLETED":
        return "请先完成个人基线采集。"
    return ""


def baseline_advisory_confirmed(state, parent=None) -> bool:
    """Round 4A-2：基线是推荐的个体参考流程，不是学习启动的绝对门槛。

    无有效基线时提示一次（本次运行内）：
    - 去采集基线 → 跳转基线页面并返回 False（不开始任务）；
    - 暂时跳过   → 允许正常开始并返回 True（记忆决定，不再重复打扰）。

    教学演示（mock）与已有有效基线（COMPLETED）直接放行。
    只做 UI 提示，不修改 Baseline 状态机、不写任何持久化数据。
    """
    if getattr(state, "mode", "live") == "mock":
        return True
    if getattr(state, "baseline_status", "IDLE") == "COMPLETED":
        return True
    if getattr(state, "_baseline_skip_prompted", False):
        return True

    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setWindowTitle("个人基线建议")
    box.setText(
        "尚未建立个人静息基线。\n"
        "完成基线后可以获得更完整的个体化学习状态对比。"
    )
    go_button = box.addButton("去采集基线", QMessageBox.AcceptRole)
    box.addButton("暂时跳过", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is go_button:
        window = parent.window() if parent is not None else None
        if window is not None and hasattr(window, "_navigate_to"):
            window._navigate_to("baseline")
        return False
    # 用户明确选择跳过：本次运行内不再重复提示。
    state._baseline_skip_prompted = True
    return True

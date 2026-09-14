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

"""Round 4A：学生单次学习报告构建器（纯读取）。

职责：
- 输入：SessionRecord + 可选 BaselineResultStore 摘要 dict；
- 输出：只读 report dict，供报告视图渲染。

边界与约束：
- 只读取已持久化字段（session.json 的 tasks / events /
  quality_summary / probability_summary / avg_attention / avg_meditation
  与 BaselineResultStore 最新摘要），不读取大 CSV、不绘制趋势曲线；
- 不修改 SessionRecord / TaskRecord / EventMarker / History / Baseline，
- 不新增任何持久化字段；
- 所有缺失值统一显示“暂无数据”，绝不用 0 / 0% / None / NaN 兜底；
- 状态倾向仅为“本次会话的状态统计摘要”，不是医学情绪诊断
  （AGENTS.md 第2节科学边界）；
- AI 建议只转述历史事件中真实存在的 label / note（Round 2H：
  AI 只提供建议，不执行难度调整），不伪造原始建议全文。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from services.dashboard_state import (
    CLASS_DISPLAY,
    DIFFICULTY_DISPLAY,
    EventMarker,
    SessionRecord,
    TaskRecord,
)

MISSING = "暂无数据"
MISSING_FEEDBACK = "本次未填写主观反馈"
MISSING_AI = "本次未触发额外学习建议"
MISSING_BASELINE = "暂无可用基线对比"

SOURCE_DISPLAY = {"live": "实时采集", "mock": "教学演示", "replay": "离线数据回放"}
EVENT_SOURCE_DISPLAY = {
    "system": "系统记录",
    "teacher": "教师观察",
    "self_report": "学生反馈",
    # live/mock/replay 前缀的系统事件同样属于机器记录
    "live": "系统记录",
    "mock": "系统记录",
    "replay": "系统记录",
}
TASK_STATUS_DISPLAY = {"running": "进行中", "completed": "已完成"}
AI_EVENT_TYPES = {"intervention", "ai_state_change"}
SELF_REPORT_TYPES = {"self_report"}
STATE_SUMMARY_NOTE = "本次会话的状态统计摘要（非医学情绪诊断）"

STABLE_SUMMARY = "本次学习过程整体较稳定，可保持当前学习节奏。"
LOAD_SUMMARY = (
    "本次学习过程中出现阶段性专注下降或学习负荷偏高，"
    "后续可适当调整连续学习时间或学习节奏。"
)
NO_DATA_SUMMARY = "本次记录已完成，暂无足够数据生成进一步总结。"


def _num(value: Any) -> Optional[float]:
    """把持久化值安全转换为 float；非数值（None/空串/bool）返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _duration_text(seconds: float) -> str:
    # 统一秒数显示规则：截断到已完整经过的秒数（与任务页/History 一致），
    # 不四舍五入，避免 97.9s 在不同页面出现 1分37秒 / 1分38秒 的 +1 秒误差。
    total = max(0, int(float(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}分{secs}秒"


def _fmt_time(value: str, timestamp: float) -> str:
    text = str(value or "").strip()
    if text:
        text = text.replace("T", " ").split("+")[0].strip()
        return text[:19]
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(timestamp)))
    except (TypeError, ValueError, OSError):
        return MISSING


def _metric_section(value: Any) -> dict:
    """Attention / Meditation 单指标：0 表示无有效样本（finalize 空历史得 0.0）。"""
    number = _num(value)
    available = number is not None and number > 0
    return {
        "available": available,
        "value": number if available else None,
        "text": f"{number:.1f}" if available else MISSING,
    }


def _build_state_summary(record: SessionRecord) -> dict:
    summary = dict(record.probability_summary or {})
    sample_count = _num(summary.get("sample_count")) or 0.0
    available = sample_count > 0

    def _pct(key: str) -> str:
        number = _num(summary.get(key))
        return f"{number * 100:.1f}%" if number is not None else MISSING

    def _state(key: str) -> str:
        state = str(summary.get(key) or "").strip()
        return CLASS_DISPLAY.get(state, state) if state else MISSING

    return {
        "available": available,
        "sample_count": int(sample_count),
        "mean_positive_text": _pct("mean_positive"),
        "mean_neutral_text": _pct("mean_neutral"),
        "mean_negative_text": _pct("mean_negative"),
        "dominant_state_text": _state("dominant_state"),
        "final_state_text": _state("final_state"),
        "note": STATE_SUMMARY_NOTE,
        "empty_text": MISSING,
    }


def _build_quality(record: SessionRecord) -> dict:
    summary = dict(record.quality_summary or {})
    sample_count = _num(summary.get("sample_count")) or 0.0
    usable = _num(summary.get("usable_ratio"))
    if sample_count <= 0:
        level_text = MISSING
    elif usable is None:
        level_text = MISSING
    elif usable >= 0.8:
        level_text = "良好"
    elif usable >= 0.5:
        level_text = "一般"
    else:
        level_text = "部分区间不可解释"
    return {
        "available": sample_count > 0,
        "level_text": level_text,
        "empty_text": MISSING,
    }


def _build_timeline(events: list[EventMarker]) -> list[dict]:
    ordered = sorted(events, key=lambda e: float(getattr(e, "timestamp", 0.0) or 0.0))
    items = []
    for event in ordered:
        source = str(event.source or "system")
        content = str(event.content or event.label or "").strip()
        if not content:
            continue
        items.append({
            "time": _fmt_time(event.time, event.timestamp),
            "source": source,
            "source_display": EVENT_SOURCE_DISPLAY.get(source, source or "系统记录"),
            "content": content,
            "note": str(event.note or "").strip(),
        })
    return items


def _build_self_feedback(events: list[EventMarker]) -> dict:
    feedback = [
        e for e in events
        if e.type in SELF_REPORT_TYPES or e.source in SELF_REPORT_TYPES
    ]
    feedback = sorted(feedback, key=lambda e: float(e.timestamp or 0.0))
    items = [{
        "time": _fmt_time(e.time, e.timestamp),
        "text": str(e.content or e.label or "").strip(),
        "note": str(e.note or "").strip(),
    } for e in feedback]
    return {"available": bool(items), "items": items, "empty_text": MISSING_FEEDBACK}


def _build_ai_advice(events: list[EventMarker]) -> dict:
    # 完整 adaptive_feedback_text 未持久化，只能转述事件中真实存在的
    # label / note；严禁补写或改写为“已执行难度调整”等表述。
    related = [
        e for e in events
        if e.type in AI_EVENT_TYPES or e.category == "intervention"
    ]
    related = sorted(related, key=lambda e: float(e.timestamp or 0.0))
    items = [{
        "time": _fmt_time(e.time, e.timestamp),
        "label": str(e.content or e.label or "").strip(),
        "note": str(e.note or "").strip(),
    } for e in related]
    return {"available": bool(items), "items": items, "empty_text": MISSING_AI}


def _build_baseline(attention: dict, meditation: dict, baseline: Any) -> dict:
    lines = []
    if isinstance(baseline, dict) and baseline:
        base_att = _num(baseline.get("avg_attention"))
        base_med = _num(baseline.get("avg_meditation"))
        # 只做绝对值对照，不计算百分比提升，不画曲线。
        if attention["available"] and base_att is not None:
            lines.append(
                f"本次平均专注度：{attention['value']:.1f}　"
                f"个人基线平均专注度：{base_att:.1f}"
            )
        if meditation["available"] and base_med is not None:
            lines.append(
                f"本次平均放松度：{meditation['value']:.1f}　"
                f"个人基线平均放松度：{base_med:.1f}"
            )
    return {"available": bool(lines), "lines": lines, "empty_text": MISSING_BASELINE}


def _build_summary_text(events: list[EventMarker], state_available: bool,
                        attention: dict, meditation: dict, quality: dict) -> str:
    has_intervention = any(
        e.type == "intervention" or e.category == "intervention" for e in events
    )
    if has_intervention:
        return LOAD_SUMMARY
    has_interpretive = (
        state_available or attention["available"] or meditation["available"]
    )
    # 质量差（大量区间不可解释）时不作“整体稳定”的结论。
    quality_ok = quality["available"] and quality["level_text"] in {"良好", "一般"}
    if has_interpretive and quality_ok:
        return STABLE_SUMMARY
    if has_interpretive and not quality["available"]:
        return STABLE_SUMMARY
    return NO_DATA_SUMMARY


def build_learning_report(record: SessionRecord, baseline: Any = None) -> dict:
    """构建学生单次学习报告（纯读取，无副作用）。

    Args:
        record: 已持久化的 SessionRecord（含 tasks / events / 汇总字段）。
        baseline: BaselineResultStore.get(student_id) 的最新摘要 dict 或 None。

    Returns:
        只读 report dict；所有缺失字段统一为“暂无数据”类文案。
    """
    tasks = [t for t in (record.tasks or []) if isinstance(t, TaskRecord)]
    events = [e for e in (record.events or []) if isinstance(e, EventMarker)]

    is_demo = bool(getattr(record, "demo", False)) or record.source == "mock"
    source_display = SOURCE_DISPLAY.get(record.source, record.source or MISSING)

    task_rows = [{
        "name": t.name or "自由学习",
        "difficulty": t.difficulty,
        "difficulty_display": DIFFICULTY_DISPLAY.get(
            t.difficulty, t.difficulty or MISSING
        ),
        # 任务有效时长只读 TaskRecord.duration_seconds，
        # 绝不用 end_time - start_time 重新计算（暂停时间已被排除）。
        "duration_seconds": float(t.duration_seconds or 0.0),
        "duration_text": _duration_text(t.duration_seconds or 0.0),
        "status_display": TASK_STATUS_DISPLAY.get(t.status, t.status or MISSING),
    } for t in tasks]

    attention = _metric_section(getattr(record, "avg_attention", None))
    meditation = _metric_section(getattr(record, "avg_meditation", None))
    state_summary = _build_state_summary(record)
    quality = _build_quality(record)
    timeline = _build_timeline(events)
    self_feedback = _build_self_feedback(events)
    ai_advice = _build_ai_advice(events)
    baseline_section = _build_baseline(attention, meditation, baseline)
    summary_text = _build_summary_text(
        events, state_summary["available"], attention, meditation, quality
    )

    student_name = str(record.user_name or "").strip()
    if not student_name:
        student_name = str(record.user_id or "").strip() or "未知学生"

    return {
        "session_id": str(record.session_id or ""),
        "student_name": student_name,
        "student_id": str(record.user_id or ""),
        "date": _fmt_time(record.start_time, 0.0),
        "source": str(record.source or ""),
        "source_display": source_display,
        "is_demo": is_demo,
        "status": str(record.status or ""),
        "duration_seconds": float(record.duration_seconds or 0.0),
        "duration_text": _duration_text(record.duration_seconds or 0.0),
        "tasks": task_rows,
        "has_tasks": bool(task_rows),
        "session_notes": str(record.notes or "").strip(),
        "attention": attention,
        "meditation": meditation,
        "state_summary": state_summary,
        "quality": quality,
        "timeline": timeline,
        "self_feedback": self_feedback,
        "ai_advice": ai_advice,
        "baseline": baseline_section,
        "summary_text": summary_text,
    }

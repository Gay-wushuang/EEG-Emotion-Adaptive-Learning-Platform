"""Round 4B：教师端"学生近期学习概览"（纯读取统计）。

职责：
- 输入：同一学生的 SessionRecord 列表 + 数据来源（live / mock）；
- 输出：只读 summary dict（近期记录表 / 近期统计 / 事件时间观察 /
  近期表现总结 / 后续教学参考）。

边界与约束（需求冻结）：
- 只消费已持久化 History 字段（SessionRecord / TaskRecord / EventMarker /
  avg_attention / avg_meditation / duration_seconds / probability_summary）；
- 禁止读取 Raw EEG / 大 CSV / 重跑模型 / 重推理；不新增任何脑电指标；
- 缺失值统一"暂无数据"，绝不把缺失 Attention 当 0 参与平均；
- Demo 与 Live 严格分开统计（source 参数二选一，不提供混合模式）；
- 长期规律（事件出现时间）要求 ≥3 次有效历史、≥3 个事件相对时间样本
  且来自 ≥2 个不同会话，禁止单次 Session 生成长期结论；
- 教学参考仅规则式措辞（"建议 / 参考 / 可考虑"），不作诊断表述
  （AGENTS.md 第2节科学边界）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from services.dashboard_state import (
    EventMarker,
    SessionRecord,
    TaskRecord,
)

MISSING = "暂无数据"
TIMING_INSUFFICIENT = "当前历史记录不足以判断需要关注事件的稳定出现时间。"
SUMMARY_FEW_RECORDS = "近期有效学习记录较少，暂不足以形成稳定趋势判断。"
SUMMARY_STABLE = "近期学习过程整体较稳定。"
SUMMARY_LOAD = (
    "近期多次出现学习负荷相关提示，建议关注连续学习时长和任务节奏。"
)
SUMMARY_DISTRACTED = (
    "近期教师观察中多次出现走神相关记录，建议增加课堂互动或阶段性提问。"
)

SOURCE_DISPLAY = {"live": "实时采集", "mock": "教学演示", "replay": "离线数据回放"}
DEFAULT_RECENT_COUNT = 5
MAX_RECENT_COUNT = 10

# "高疲劳"自我反馈 / "走神"教师观察的文本关键词（仅用于已有事件文本的
# 粗分类，不构成新的生理指标定义）。
FATIGUE_KEYWORDS = ("疲劳", "疲惫", "很累", "有点累", "太累", "犯困", "困")
DISTRACTED_KEYWORDS = ("走神", "分心", "注意力不集中", "不专注")
LOAD_KEYWORDS = ("负荷",)

# 事件相对任务开始时间的合理窗口（分钟）：窗口外视为解析噪声丢弃。
_MIN_EVENT_OFFSET_MIN = 0
_MAX_EVENT_OFFSET_MIN = 240
# 生成长期时间规律的最少样本要求。
_MIN_TIMING_SESSIONS = 3
_MIN_TIMING_SAMPLES = 3
_MIN_TIMING_DISTINCT_SESSIONS = 2


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _epoch(text: str) -> Optional[float]:
    """把 ISO / 本地时间字符串解析为 epoch；失败返回 None。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace(" ", "T")).timestamp()
    except (ValueError, OSError):
        return None


def _record_epoch(record: SessionRecord) -> float:
    """记录排序时间：优先 start_time，解析失败回退 0。"""
    return _epoch(getattr(record, "start_time", "")) or 0.0


def _duration_text(seconds: float) -> str:
    # 与任务页 / History / 学习报告一致的截断秒数规则（不四舍五入）。
    total = max(0, int(float(seconds or 0.0)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}分{secs}秒"


def _task_summary(record: SessionRecord) -> str:
    """任务列摘要：≤2 个任务逐个列出，更多则显示数量。"""
    names = [t.name for t in (record.tasks or []) if getattr(t, "name", "")]
    if not names:
        return "自由学习"
    if len(names) <= 2:
        return " + ".join(names)
    return f"{len(names)}项任务"


def _is_intervention(event: EventMarker) -> bool:
    return event.type == "intervention" or event.category == "intervention"


def _is_teacher_observation(event: EventMarker) -> bool:
    return event.type == "teacher_observation" or event.source == "teacher"


def _is_self_report(event: EventMarker) -> bool:
    return event.type == "self_report" or event.source == "self_report"


def _is_load_advice(event: EventMarker) -> bool:
    if not _is_intervention(event):
        return False
    text = f"{event.content}{event.label}{event.note}"
    return any(k in text for k in LOAD_KEYWORDS)


def _is_distracted_observation(event: EventMarker) -> bool:
    if not _is_teacher_observation(event):
        return False
    text = f"{event.content}{event.label}{event.note}"
    return any(k in text for k in DISTRACTED_KEYWORDS)


def _is_fatigue_self_report(event: EventMarker) -> bool:
    if not _is_self_report(event):
        return False
    text = f"{event.content}{event.label}{event.note}"
    return any(k in text for k in FATIGUE_KEYWORDS)


def _attention_metric(value: Any) -> dict:
    """单次 Attention：0 表示无有效样本（finalize 空历史得 0.0）。"""
    number = _num(value)
    available = number is not None and number > 0
    return {
        "available": available,
        "value": number if available else None,
        "text": f"{number:.1f}" if available else MISSING,
    }


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _avg_duration_text(values: List[float]) -> dict:
    mean = _mean(values)
    return {
        "available": mean is not None,
        "value": mean,
        "text": _duration_text(mean) if mean is not None else MISSING,
    }


def _event_minute_offset(event: EventMarker, record: SessionRecord) -> Optional[float]:
    """事件相对其所属任务开始时间的分钟数；无法可靠取得时返回 None。"""
    task_id = str(getattr(event, "task_id", "") or "")
    tasks = list(record.tasks or [])
    task = None
    if task_id:
        task = next((t for t in tasks if t.task_id == task_id), None)
    if task is None and len(tasks) == 1:
        # 单任务会话：事件没有 task_id 时回退到唯一任务。
        task = tasks[0]
    if task is None:
        return None
    task_epoch = _epoch(task.start_time)
    if task_epoch is None:
        return None
    offset = (float(event.timestamp) - task_epoch) / 60.0
    if not (_MIN_EVENT_OFFSET_MIN <= offset <= _MAX_EVENT_OFFSET_MIN):
        return None
    return offset


def _needs_attention_event(event: EventMarker) -> bool:
    return (
        _is_intervention(event)
        or _is_teacher_observation(event)
        or _is_fatigue_self_report(event)
    )


def build_recent_summary(
    records: List[SessionRecord],
    *,
    source: str = "live",
    max_sessions: int = DEFAULT_RECENT_COUNT,
) -> dict:
    """构建学生近期学习概览（纯读取，无副作用）。

    Args:
        records: 学生全部历史 SessionRecord（顺序不限）。
        source: "live"（只统计实时采集）或 "mock"（只统计教学演示）。
            不提供混合模式——Demo 与 Live 不合并为同一个长期趋势。
        max_sessions: 统计最近 N 次，默认 5，上限 10。

    Returns:
        只读 summary dict；缺失值统一为"暂无数据"类文案。
    """
    if source not in ("live", "mock"):
        raise ValueError("source must be 'live' or 'mock' (demo/live never mix)")
    count = max(1, min(int(max_sessions), MAX_RECENT_COUNT))

    def _eligible(record: SessionRecord) -> bool:
        is_demo = bool(getattr(record, "demo", False)) or record.source == "mock"
        return is_demo if source == "mock" else (not is_demo and record.source == "live")

    recent = sorted(
        (r for r in records if _eligible(r)),
        key=_record_epoch,
        reverse=True,
    )[:count]

    rows = []
    attention_values: List[float] = []
    duration_values: List[float] = []
    ai_advice_total = teacher_obs_total = self_report_total = 0
    load_advice_total = distracted_total = fatigue_report_total = 0
    timing_offsets: List[float] = []
    timing_sessions: set = set()

    for record in recent:
        events = [e for e in (record.events or []) if isinstance(e, EventMarker)]
        ai_count = sum(1 for e in events if _is_intervention(e))
        teacher_count = sum(1 for e in events if _is_teacher_observation(e))
        report_count = sum(1 for e in events if _is_self_report(e))
        ai_advice_total += ai_count
        teacher_obs_total += teacher_count
        self_report_total += report_count
        load_advice_total += sum(1 for e in events if _is_load_advice(e))
        distracted_total += sum(1 for e in events if _is_distracted_observation(e))
        fatigue_report_total += sum(1 for e in events if _is_fatigue_self_report(e))

        for event in events:
            if not _needs_attention_event(event):
                continue
            offset = _event_minute_offset(event, record)
            if offset is not None:
                timing_offsets.append(offset)
                timing_sessions.add(record.session_id)

        attention = _attention_metric(getattr(record, "avg_attention", None))
        if attention["available"]:
            attention_values.append(attention["value"])
        duration_values.append(float(record.duration_seconds or 0.0))

        rows.append({
            "session_id": str(record.session_id or ""),
            "date": str(record.start_time or "")[:19].replace("T", " ") or MISSING,
            "task_summary": _task_summary(record),
            "source_display": SOURCE_DISPLAY.get(record.source, record.source or MISSING),
            "duration_text": _duration_text(record.duration_seconds or 0.0),
            "attention": attention,
            "ai_advice_count": ai_count,
            "teacher_observation_count": teacher_count,
            "self_report_count": report_count,
        })

    # 平均 Attention：只对真实存在的数值求平均，并注明有效记录数。
    att_mean = _mean(attention_values)
    if att_mean is None:
        attention_text = MISSING
    elif len(attention_values) == len(recent):
        attention_text = f"{att_mean:.1f}"
    else:
        attention_text = f"{att_mean:.1f}（基于{len(attention_values)}次有效记录）"

    stats = {
        "count": len(recent),
        "avg_duration": _avg_duration_text(duration_values),
        "avg_attention": {
            "available": att_mean is not None,
            "value": att_mean,
            "based_on": len(attention_values),
            "text": attention_text,
        },
        "ai_advice_count": ai_advice_total,
        "teacher_observation_count": teacher_obs_total,
        "self_report_count": self_report_total,
        "load_advice_count": load_advice_total,
        "distracted_observation_count": distracted_total,
        "fatigue_report_count": fatigue_report_total,
    }

    # 需要关注事件的时间观察：仅当历史足够且样本跨会话时才生成，
    # 禁止单次 Session 生成长期规律。
    timing_available = (
        len(recent) >= _MIN_TIMING_SESSIONS
        and len(timing_offsets) >= _MIN_TIMING_SAMPLES
        and len(timing_sessions) >= _MIN_TIMING_DISTINCT_SESSIONS
    )
    if timing_available:
        low, high = int(min(timing_offsets)), int(max(timing_offsets))
        if low == high:
            timing_text = f"近期需要关注的事件多出现在连续学习约{low}分钟后。"
        else:
            timing_text = f"近期需要关注的事件多出现在连续学习约{low}～{high}分钟后。"
    else:
        timing_text = TIMING_INSUFFICIENT

    # 近期表现总结（简单规则，情况 A～D）
    if len(recent) < 3:
        performance_lines = [SUMMARY_FEW_RECORDS]
    else:
        performance_lines = []
        if load_advice_total >= 2:
            performance_lines.append(SUMMARY_LOAD)
        if distracted_total >= 2:
            performance_lines.append(SUMMARY_DISTRACTED)
        if not performance_lines:
            performance_lines.append(SUMMARY_STABLE)

    # 后续教学参考（规则式，措辞仅"建议 / 参考 / 可考虑"）
    suggestions = []
    if len(recent) >= 3:
        avg_duration = stats["avg_duration"]["value"] or 0.0
        if avg_duration >= 25 * 60 and load_advice_total >= 2:
            suggestions.append("参考近期记录：连续学习时间较长且多次出现负荷提示，可考虑缩短连续学习区间。")
        if distracted_total >= 2:
            suggestions.append("参考近期教师观察：可考虑增加课堂互动或阶段性提问。")
        if fatigue_report_total >= 2:
            suggestions.append("参考近期学生反馈：建议增加休息安排。")
        if not suggestions:
            suggestions.append("近期学习过程整体较稳定，可保持当前教学节奏。")

    return {
        "source": source,
        "is_demo": source == "mock",
        "source_display": SOURCE_DISPLAY[source],
        "max_sessions": count,
        "student_id": str(recent[0].user_id if recent else ""),
        "student_name": str(recent[0].user_name if recent else ""),
        "sessions": rows,
        "stats": stats,
        "event_timing": {
            "available": timing_available,
            "sample_count": len(timing_offsets),
            "distinct_sessions": len(timing_sessions),
            "text": timing_text,
        },
        "performance_summary": performance_lines,
        "teaching_suggestions": suggestions,
    }

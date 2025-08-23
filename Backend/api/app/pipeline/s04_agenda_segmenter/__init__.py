from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from ...schema.process_information import ProcessInformation

def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1]
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _sec_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds()

class AgendaSegmenterPipeline:
    """
    把 info.transcription（start/end/text/speaker）按时间窗口分发到
    info.customer_meeting_detail['agenda'][i]['lines']。
    窗口 = [calculatedStartTime, calculatedStartTime + lengthMinutes)
    以 meeting.startTime/meeting.date 作为 0 秒。
    """
    def process(self, info: ProcessInformation):
        detail = getattr(info, "customer_meeting_detail", None)
        lines: List[Dict[str, Any]] = getattr(info, "transcription", None)
        if not detail or not lines:
            logger.warning("AgendaSegmenter: 缺少 meeting detail 或 transcription，跳过。")
            return

        meeting_start = _parse_iso(detail.get("startTime") or detail.get("date"))
        agenda: List[Dict[str, Any]] = detail.get("agenda") or []
        if not meeting_start or not agenda:
            logger.warning("AgendaSegmenter: 缺少 meeting startTime 或 agenda，跳过。")
            return

        # 预计算每个议程窗口（相对秒）
        windows = []
        for it in agenda:
            a_start = _parse_iso(it.get("calculatedStartTime")) or meeting_start
            minutes = float(it.get("lengthMinutes") or 0)
            a_end = a_start + timedelta(minutes=minutes)
            windows.append({
                "item": it,
                "start_sec": max(0.0, _sec_between(meeting_start, a_start)),
                "end_sec":   max(0.0, _sec_between(meeting_start, a_end)),
            })
            it["lines"] = []  # 清空/创建承载字段

        # 分配每条识别行
        for ln in lines:
            s = float(ln.get("start") or 0.0)
            e = float(ln.get("end") or s)
            for w in windows:
                if (s < w["end_sec"]) and (e > w["start_sec"]):
                    w["item"]["lines"].append({
                        "start": round(s, 2),
                        "end":   round(e, 2),
                        "text":  (ln.get("text") or "").strip(),
                        "speaker": ln.get("speaker", "Unknown"),
                    })
                    break

        for w in windows:
            w["item"]["lines"].sort(key=lambda x: (x["start"], x["end"]))
            logger.info(f"Agenda '{w['item'].get('title','')}' ← {len(w['item']['lines'])} lines")

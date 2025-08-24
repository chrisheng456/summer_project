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
    # Return the time difference between two datetimes in seconds.
    return (b - a).total_seconds()

class AgendaSegmenterPipeline:
    """
    Distributes transcribed lines (with start/end times, text, and speaker info)
    into the corresponding agenda items provided in `info.customer_meeting_detail`.

    Each agenda item defines a time window:
        [calculatedStartTime, calculatedStartTime + lengthMinutes)

    Transcribed lines that fall into a window are assigned to that agenda item.
    All times are relative to the meeting start time.
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

        # Precompute time windows for each agenda item (in seconds from meeting start)
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
            it["lines"] = []

        # Assign each transcribed line to the correct agenda window
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

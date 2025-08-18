from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from ...schema.process_information import ProcessInformation

DEF_FALLBACK_MIN = 5.0  # 兜底分钟数

def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", ""))
    except Exception:
        return None

def _sec_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds()

def _parent_index(number: str) -> Optional[str]:
    # "4.1" -> "4"; "6.2" -> "6"; 顶层返回 None
    if not number or "." not in number:
        return None
    return number.rsplit(".", 1)[0]

class AgendaSegmenterPipeline:
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

        # 预处理：清空承载字段 + 建立 number -> idx 的索引，便于找到父项
        num_to_idx: Dict[str, int] = {}
        for i, it in enumerate(agenda):
            it["lines"] = []
            if it.get("number"):
                num_to_idx[str(it["number"])] = i

        # 第一次遍历：先确定“起点”
        starts: List[datetime] = [None] * len(agenda)  # type: ignore
        lengths_min: List[float] = [float(it.get("lengthMinutes") or 0.0) for it in agenda]

        for i, it in enumerate(agenda):
            cs = _parse_iso(it.get("calculatedStartTime"))
            if cs:
                starts[i] = cs
                continue

            # 缺开始时间：
            indent = int(it.get("indent", 0))
            number = str(it.get("number") or "")
            if indent > 0:
                # 子项优先继承父项开始
                pnum = _parent_index(number)
                if pnum and pnum in num_to_idx:
                    pidx = num_to_idx[pnum]
                    starts[i] = starts[pidx] or _parse_iso(agenda[pidx].get("calculatedStartTime")) or meeting_start
                else:
                    # 找不到父项，降级用“前一项结束”或会议开始
                    prev_end = None
                    if i > 0 and starts[i-1]:
                        prev_end = starts[i-1] + timedelta(minutes=lengths_min[i-1] or 0)
                    starts[i] = prev_end or meeting_start
            else:
                # 顶层项：用“前一项结束”作为开始
                prev_end = None
                if i > 0 and starts[i-1]:
                    prev_end = starts[i-1] + timedelta(minutes=lengths_min[i-1] or 0)
                starts[i] = prev_end or meeting_start

        # 第二次遍历：填充长度（缺失则用下一项的开始 - 本项开始；最后再兜底）
        for i in range(len(agenda)):
            if lengths_min[i] > 0:
                continue
            cur_st = starts[i] or meeting_start
            nxt_st = None
            for j in range(i+1, len(agenda)):
                if starts[j]:
                    nxt_st = starts[j]
                    break
            if nxt_st and nxt_st > cur_st:
                lengths_min[i] = max((_sec_between(cur_st, nxt_st) / 60.0), 0.1)
            else:
                lengths_min[i] = DEF_FALLBACK_MIN

        # 第三次遍历：对子项进行“父项窗口”裁剪
        windows = []
        for i, it in enumerate(agenda):
            st_dt = starts[i] or meeting_start
            en_dt = st_dt + timedelta(minutes=lengths_min[i])
            # 若为子项，强制限制到父项窗口
            indent = int(it.get("indent", 0))
            if indent > 0:
                pnum = _parent_index(str(it.get("number") or ""))
                if pnum and pnum in num_to_idx:
                    pidx = num_to_idx[pnum]
                    pst = starts[pidx] or meeting_start
                    pen = pst + timedelta(minutes=lengths_min[pidx])
                    if st_dt < pst: st_dt = pst
                    if en_dt > pen: en_dt = pen
            # 忽略无效窗口
            if en_dt <= st_dt:
                continue

            windows.append({
                "idx": i,
                "item": it,
                "title": it.get("title", ""),
                "indent": int(it.get("indent", 0)),
                "start_sec": max(0.0, _sec_between(meeting_start, st_dt)),
                "end_sec":   max(0.0, _sec_between(meeting_start, en_dt)),
            })

        windows.sort(key=lambda w: (w["start_sec"], w["end_sec"]))

        # 分配规则：最大重叠优先；若并列，indent 大（子项）优先，其次窗口更短优先
        def _overlap(a0, a1, b0, b1):
            return max(0.0, min(a1, b1) - max(a0, b0))

        for ln in lines:
            s = float(ln.get("start") or 0.0)
            e = float(ln.get("end") or s)
            best = None
            best_key = None  # (overlap, indent, -duration)

            for w in windows:
                ov = _overlap(s, e, w["start_sec"], w["end_sec"])
                if ov <= 0:
                    continue
                duration = w["end_sec"] - w["start_sec"]
                key = (ov, w["indent"], -duration)
                if (best_key is None) or (key > best_key):
                    best_key = key
                    best = w

            target = best or windows[-1]  # 兜底放到最后一项，避免漏行
            target["item"]["lines"].append({
                "start": round(s, 2),
                "end":   round(e, 2),
                "text":  (ln.get("text") or "").strip(),
                "speaker": ln.get("speaker", "Unknown"),
            })

        for w in windows:
            w["item"]["lines"].sort(key=lambda x: (x["start"], x["end"]))
            logger.info(f"Agenda '{w['title']}' [{w['start_sec']:.0f}s~{w['end_sec']:.0f}s, indent={w['indent']}] "
                        f"← {len(w['item']['lines'])} lines")

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import io, re
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Flowable
)

# ---------- 文本处理 ----------
_ZWS = "\u200b"
_SOFT_WRAP_EVERY = 24

def _soft_wrap(text: str, every: int = _SOFT_WRAP_EVERY) -> str:
    def wrap_token(tok: str) -> str:
        if len(tok) <= every or any(ch.isspace() for ch in tok):
            return tok
        return _ZWS.join(tok[i:i+every] for i in range(0, len(tok), every))
    text = (text or "").replace("\u00A0"," ").replace("\u2009"," ").replace("\u202F"," ").replace("\u2060","")
    parts = re.findall(r"\S+|\s+", text, flags=re.UNICODE)
    return "".join(wrap_token(p) if not p.isspace() else p for p in parts)

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[^\S\n\t]+"," ", s)
    s = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]","", s)
    return _soft_wrap(s).strip()

def _fmt_time(v: Any) -> str:
    if v is None: return ""
    try:
        x = float(v); m = int(x // 60); s = int(round(x - m*60))
        return f"{m:02d}:{s:02d}"
    except Exception:
        return _norm(str(v))

# ---------- 样式 ----------
def _style_title() -> ParagraphStyle:
    return ParagraphStyle("TitleH1", parent=getSampleStyleSheet()["Heading1"],
        fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10, wordWrap="CJK")

def _style_h2() -> ParagraphStyle:
    return ParagraphStyle("H2", parent=getSampleStyleSheet()["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6, wordWrap="CJK")

def _style_h3() -> ParagraphStyle:
    return ParagraphStyle("H3", parent=getSampleStyleSheet()["Heading3"],
        fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=6, spaceAfter=4, wordWrap="CJK")

def _style_body(size: int = 11) -> ParagraphStyle:
    return ParagraphStyle("Body", parent=getSampleStyleSheet()["BodyText"],
        fontName="Helvetica", fontSize=size, leading=max(14, int(size*1.3)), spaceAfter=3, wordWrap="CJK")

def _p(text: str, size: int = 11) -> Paragraph:
    return Paragraph(_norm(text), _style_body(size))

# ---------- 安全表格 ----------
def _safe_table(data: List[List[Any]], col_widths=None, repeat_header: bool = False) -> Flowable:
    """
    任何时候只要 data 为空或列数为 0，就返回一个小 Spacer，避免 Table(0x0) 的报错。
    """
    try:
        if not data or not isinstance(data, list):
            return Spacer(1, 0.01*cm)
        if not data[0] or len(data[0]) == 0:
            return Spacer(1, 0.01*cm)
        tbl = Table(data, colWidths=col_widths, repeatRows=(1 if repeat_header else 0))
        return tbl
    except Exception:
        return Spacer(1, 0.01*cm)

# ---------- 组件 ----------
def _kv_table(rows: List[Tuple[str,str]], col_widths: Optional[List[float]] = None) -> Flowable:
    if not rows:
        return Spacer(1, 0.01*cm)
    data = [[_p(f"<b>{_norm(k)}</b>"), _p(_norm(v))] for k, v in rows]
    tbl = _safe_table(data, col_widths)
    if isinstance(tbl, Table):
        tbl.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.lightgrey),
            ("BOX",(0,0),(-1,-1),0.25,colors.lightgrey),
            ("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),
        ]))
    return tbl

def _agenda_lines_table(lines: List[Dict[str,Any]]) -> Flowable:
    header = [_p("<b>Start</b>",10), _p("<b>End</b>",10), _p("<b>Speaker</b>",10), _p("<b>Text</b>",10)]
    data: List[List[Any]] = [header]
    for ln in (lines or []):
        data.append([
            _p(_fmt_time(ln.get("start")),10),
            _p(_fmt_time(ln.get("end")),10),
            _p(_norm(ln.get("speaker","")),10),
            _p(_norm(ln.get("text","")),10),
        ])
    tbl = _safe_table(data, [2.0*cm, 2.0*cm, 4.0*cm, 11.0*cm], repeat_header=True)
    if isinstance(tbl, Table):
        tbl.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F0F3F7")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
    return tbl

def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9); canvas.setFillColor(colors.grey)
    canvas.drawRightString(20.0*cm, 1.35*cm, f"Page {doc.page}")
    canvas.drawString(2*cm, 1.35*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    canvas.restoreState()

# ---------- 主体 ----------
def build_meeting_pdf(meeting_detail: Dict[str,Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.6*cm, bottomMargin=1.8*cm,
        title=_norm(meeting_detail.get("name") or "Meeting"),
        author="summer_project"
    )
    story: List[Any] = []

    # 顶部
    story.append(Paragraph(_norm(meeting_detail.get("name","Meeting")), _style_title()))
    top_rows: List[Tuple[str,str]] = [
        ("Meeting ID", str(meeting_detail.get("id",""))),
        ("Date",       _norm(meeting_detail.get("date") or "")),
        ("Start Time", _norm(meeting_detail.get("startTime") or "")),
        ("Location",   _norm(meeting_detail.get("location") or "")),
    ]
    attendees = meeting_detail.get("attendees") or []
    if attendees:
        top_rows.append(("Attendees", ", ".join(_norm(x.get("name","")) for x in attendees)))
    story.append(_kv_table(top_rows, [4.0*cm, 11.0*cm]))
    story.append(Spacer(1, 0.4*cm))

    agenda: List[Dict[str,Any]] = meeting_detail.get("agenda") or []
    if not agenda:
        story.append(_p("No agenda items."))

    for idx, it in enumerate(agenda):
        if idx > 0:
            story.append(Spacer(1, 0.4*cm))
        number = str(it.get("number",""))
        title  = _norm(it.get("title",""))
        story.append(Paragraph(f"Agenda {number}: {title}", _style_h2()))

        meta_rows: List[Tuple[str,str]] = []
        if it.get("owner"): meta_rows.append(("Owner", _norm(it.get("owner"))))
        if it.get("calculatedStartTime"): meta_rows.append(("Start", _norm(it.get("calculatedStartTime"))))
        if it.get("lengthMinutes") is not None: meta_rows.append(("Duration", f"{it.get('lengthMinutes')} min"))
        if it.get("label"): meta_rows.append(("Label", f"{_norm(it.get('label'))} (score: {it.get('label_score', '')})"))

        story.append(_kv_table(meta_rows, [3.0*cm, 12.0*cm]))

        if it.get("explanation"):
            story.append(Paragraph("<b>Explanation</b>", _style_h3()))
            story.append(_p(_norm(it.get("explanation"))))
        if it.get("summary"):
            story.append(Paragraph("<b>Summary</b>", _style_h3()))
            story.append(_p(_norm(it.get("summary"))))

        lines = it.get("lines") or []
        if lines:
            story.append(Paragraph("<b>Lines</b>", _style_h3()))
            story.append(_agenda_lines_table(lines))
        else:
            story.append(_p("No lines."))

        if idx % 3 == 2:
            story.append(PageBreak())

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buf.getvalue()

def export_pdf_from_payload(payload: Dict[str,Any]) -> bytes:
    if not payload: raise ValueError("empty payload")
    md = payload.get("customer_meeting_detail") or payload.get("meeting_detail") or payload
    if not isinstance(md, dict) or not md.get("agenda"):
        raise ValueError("not a valid meeting detail JSON (missing 'agenda')")
    return build_meeting_pdf(md)

# 兼容旧名
export_to_pdf = export_pdf_from_payload

from fpdf import FPDF
import json
from typing import Union
from datetime import datetime


class PDFWithFooter(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def format_time(timestr):
    if not timestr:
        return ""
    try:
        if timestr.endswith("Z"):
            timestr = timestr[:-1]
        dt = datetime.fromisoformat(timestr)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return timestr


def export_to_pdf(result_json: Union[str, dict]) -> bytes:
    if isinstance(result_json, str):
        data = json.loads(result_json)
    else:
        data = result_json
    pdf = PDFWithFooter()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(40, 40, 120)
    pdf.cell(0, 12, data.get("name", "Meeting Minutes"), ln=True, align="C")
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # 会议基本信息
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0)
    for k in ["date", "startTime", "location"]:
        v = data.get(k, "")
        if k in ("date", "startTime"):
            v = format_time(v)
        if v:
            pdf.cell(0, 8, f"{k.title()}: {v}", ln=True)
    pdf.ln(2)

    # 与会者
    attendees = data.get("attendees", [])
    if attendees:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(40, 80, 120)
        pdf.cell(0, 9, "Attendees:", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0)
        for att in attendees:
            pdf.cell(8)
            pdf.cell(0, 7, f"- {att.get('name', '')}", ln=True)
        pdf.ln(2)

    # 议程
    agenda = data.get("agenda", [])
    for idx, item in enumerate(agenda, 1):
        # 议题分块背景色
        y0 = pdf.get_y()
        pdf.set_fill_color(230, 240, 255)
        pdf.rect(10, y0, 190, 10, "F")
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(0, 60, 120)
        title = f"Agenda {item.get('number', idx)}: {item.get('title', '')}"
        pdf.set_y(y0)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0)
        if item.get("owner"):
            pdf.cell(10)
            pdf.cell(0, 7, f"Owner: {item['owner']}", ln=True)
        if item.get("calculatedStartTime"):
            pdf.cell(10)
            pdf.cell(
                0,
                7,
                f"Start: {format_time(item['calculatedStartTime'])}",
                ln=True,
            )
        if item.get("lengthMinutes"):
            pdf.cell(10)
            pdf.cell(0, 7, f"Duration: {item['lengthMinutes']} min", ln=True)
        if item.get("label"):
            pdf.cell(
                0,
                7,
                f"Label: {item['label']} (score: {item.get('label_score', '')})",
                ln=True,
            )
        if item.get("explanation"):
            pdf.cell(10)
            pdf.set_font("Arial", "I", 11)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 7, f"Explanation: {item['explanation']}")
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0)
        if item.get("summary"):
            pdf.cell(10)
            pdf.set_font("Arial", "B", 11)
            pdf.set_text_color(0, 100, 0)
            pdf.multi_cell(0, 7, f"Summary: {item['summary']}")
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0)
        pdf.ln(1)
        # lines
        lines = item.get("lines", [])
        for ln in lines:
            speaker = ln.get("speaker", "")
            start = ln.get("start", "")
            end = ln.get("end", "")
            text = ln.get("text", "")
            pdf.cell(15)
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"{speaker} [{start:.2f}-{end:.2f}s]:", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(0)
            pdf.cell(20)
            pdf.multi_cell(0, 6, text)
        pdf.ln(2)
        # 分隔线
        pdf.set_draw_color(200, 200, 200)
        y = pdf.get_y()
        pdf.line(10, y, 200, y)
        pdf.ln(2)

    return pdf.output(dest="S").encode("latin1")

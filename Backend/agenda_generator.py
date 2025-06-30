#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from docx import Document

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def compute_speaker_times(diarized_records):
    segs = defaultdict(list)
    for r in diarized_records:
        spk = r.get("speaker", "Unknown")
        s, e = r.get("start", 0.0), r.get("end", 0.0)
        segs[spk].append((s, e - s))
    out = {}
    for spk, lst in segs.items():
        valid = [(s, d) for s, d in lst if s > 0]
        starts, durs = zip(*valid) if valid else zip(*lst)
        out[spk] = {"start_time": min(starts), "duration": sum(durs)}
    return out

def fmt_time(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))

def build_agenda_by_speaker(summaries, time_info):
    grp = defaultdict(list)
    for x in summaries:
        grp[x["speaker"]].append(x)
    sections = []
    for spk, recs in grp.items():
        ti = time_info.get(spk, {"start_time": 0, "duration": 0})
        sec = {
            "speaker":    spk,
            "start_time": fmt_time(ti["start_time"]),
            "duration":   fmt_time(ti["duration"]),
            "items":      []
        }
        for i, x in enumerate(recs, 1):
            sec["items"].append({
                "id":        i,
                "actions":   x.get("actions", []),
                "decisions": x.get("decisions", []),
                "conflicts": x.get("conflicts", [])
            })
        sections.append(sec)
    return sections

def generate_meeting_title(summaries):
    """
    直接取第一条 summary，去掉开头的 SPEAKER_xxx 前缀，
    截到第一个分号前作为会议题目
    """
    if not summaries:
        return "会议摘要"
    raw = summaries[0].get("summary", "")
    # 去掉类似 "SPEAKER_05 " 的前缀
    title = re.sub(r'^SPEAKER_[^\s]+\s*', '', raw)
    # 截到第一个分号前
    if ";" in title:
        title = title.split(";", 1)[0]
    title = title.strip()
    # 限制长度
    if len(title) > 60:
        title = title[:60].rstrip() + "…"
    return title

def save_agenda_json(sections, title, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title,
        "agenda_by_speaker": sections
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def save_agenda_docx(sections, title, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    # 一级标题：只展示会议题目
    doc.add_heading(title, level=1)
    # 二级标题：议程
    doc.add_heading("会议议程（按说话人分段）", level=2)
    for idx, sec in enumerate(sections, 1):
        h = doc.add_heading(level=3)
        h.add_run(f"{idx}. {sec['speaker']} ").bold = True
        h.add_run(f"[{sec['start_time']} | {sec['duration']}]")
        for item in sec["items"]:
            p = doc.add_paragraph(style="List Number")
            p.add_run(f"{item['id']}. {sec['speaker']}").bold = True
            for act in item.get("actions", []):
                pa = doc.add_paragraph(style="List Bullet")
                pa.add_run(act)
            for dec in item.get("decisions", []):
                pd = doc.add_paragraph(style="List Bullet")
                pd.add_run(dec)
            for cf in item.get("conflicts", []):
                pc = doc.add_paragraph(style="List Bullet")
                pc.add_run(cf)
    doc.save(str(out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("生成简短会议题目并按说话人分段输出议程")
    base = Path(__file__).parent
    parser.add_argument("--diarized",
        default=str(base/"dataset"/"ICSI"/"diarized_json"/"Bed002_diarized.json"))
    parser.add_argument("--summary",
        default=str(base/"dataset"/"ICSI"/"after_json"/"Bed002_summary.json"))
    parser.add_argument("--out_json",
        default=str(base/"output"/"agenda.json"))
    parser.add_argument("--out_docx",
        default=str(base/"output"/"agenda.docx"))
    args = parser.parse_args()

    diar = load_json(Path(args.diarized))
    summ = load_json(Path(args.summary))
    times = compute_speaker_times(diar)
    secs  = build_agenda_by_speaker(summ, times)
    title = generate_meeting_title(summ)

    save_agenda_json(secs, title, Path(args.out_json))
    save_agenda_docx(secs, title, Path(args.out_docx))

    print("✅ 已生成简短题目及议程：")
    print("   JSON →", args.out_json)
    print("   DOCX →", args.out_docx)

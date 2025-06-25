import json
from datetime import timedelta
from pathlib import Path

from transformers import pipeline   # 或者你喜欢的任何摘要模型
from docx import Document

def load_transcript(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["lines"]  # 每项含 speaker, offset (100ns), duration (100ns), start_time, text

def segment_lines(lines, max_segment_duration_sec=300):
    segments = []
    cur_seg = {"lines": [], "start_offset": None, "end_offset": None}
    prev_speaker = None

    for ln in lines:

        if cur_seg["start_offset"] is None:
            cur_seg["start_offset"] = ln["offset"]


        elapsed_sec = (ln["offset"] - cur_seg["start_offset"]) / 10_000_000


        time_exceeded = elapsed_sec > max_segment_duration_sec
        speaker_changed = (prev_speaker is not None and ln["speaker"] != prev_speaker)

        if time_exceeded or speaker_changed:

            segments.append(cur_seg)

            cur_seg = {
                "lines": [],
                "start_offset": ln["offset"],
                "end_offset": None
            }


        cur_seg["lines"].append(ln)
        cur_seg["end_offset"] = ln["offset"] + ln["duration"]

        prev_speaker = ln["speaker"]


    if cur_seg["lines"]:
        segments.append(cur_seg)

    return segments


def extract_topic(text: str):

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    res = summarizer(text, max_length=20, min_length=5, do_sample=False)
    return res[0]["summary_text"]

def build_agenda_items(segments):

    items = []
    for idx, seg in enumerate(segments, start=1):
        # 计算起止时间
        start_off = seg["start_offset"] // 10_000_000
        end_off   = seg["end_offset"]   // 10_000_000
        start_ts  = str(timedelta(seconds=int(start_off)))
        duration  = end_off - start_off


        speakers = [ln["speaker"] for ln in seg["lines"]]
        main_spk = max(set(speakers), key=speakers.count)


        text = " ".join(ln["text"] for ln in seg["lines"])
        topic = extract_topic(text)

        items.append({
            "id": idx,
            "start_time": start_ts,
            "duration_sec": duration,
            "speaker": main_spk,
            "topic": topic
        })
    return items

def save_agenda_json(items, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"agenda": items}, f, ensure_ascii=False, indent=2)

def save_agenda_docx(items, out_path: Path):
    doc = Document()
    doc.add_heading("会议议程 (Agenda)", level=1)
    for it in items:
        p = doc.add_paragraph()
        p.add_run(f"{it['id']}. [{it['start_time']}] ").bold = True
        p.add_run(f"{it['speaker']} (时长 {it['duration_sec']} 秒): ")
        p.add_run(it["topic"])
    doc.save(out_path)

if __name__ == "__main__":
    # 1. 自动找到最新转录 JSON
    latest = sorted(Path(".").glob("meeting_minutes_*.json"))[-1]
    lines = load_transcript(latest)
    # 2. 分段
    segs = segment_lines(lines, max_segment_duration_sec=300)
    # 3. 生成 agenda item 列表
    agenda = build_agenda_items(segs)
    # 4. 输出 JSON 和 DOCX
    save_agenda_json(agenda, Path("agenda.json"))
    save_agenda_docx(agenda,   Path("agenda.docx"))
    print("✅ 议程已生成：agenda.json & agenda.docx")

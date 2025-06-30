import os
import re
import json
import spacy
from collections import defaultdict
from typing import List, Dict

# 加载 spaCy 英文模型
nlp = spacy.load("en_core_web_md")

def extract_key_elements(text: str) -> dict:
    """使用 spaCy 提取 actions/decisions/conflicts"""
    doc = nlp(text)

    def is_noise_sentence(t: str) -> bool:
        words = t.strip().split()
        if len(words) < 4 and re.fullmatch(r'([A-Za-z]\s*){1,4}', t):
            return True
        return False

    filtered_sents = [sent for sent in doc.sents if not is_noise_sentence(sent.text)]

    actions, decisions, conflicts = [], [], []

    decision_keywords = ["decide", "choose", "select", "option", "prefer", "go with", "agree", "conclude"]
    conflict_keywords = ["but", "however", "although", "though", "disagree", "problem", "issue", "concern", "doubt"]

    for sent in filtered_sents:
        text = " ".join(t.text for t in sent if not t.is_punct)
        tokens = list(sent)

        # Actions
        for token in tokens:
            if token.dep_ == "ROOT" and token.tag_ in ("MD", "VB"):
                if any(ch.dep_ == "dobj" for ch in token.children):
                    actions.append(text)
                    break

        # Decisions
        low = sent.text.lower()
        if any(k in low for k in decision_keywords):
            decisions.append(text)

        # Conflicts
        if any(k in low for k in conflict_keywords):
            conflicts.append(text)

    return {
        "actions":   actions   or ["No specific actions identified"],
        "decisions": decisions or ["No explicit decisions found"],
        "conflicts": conflicts or ["No conflicts mentioned"]
    }

def process_speaker_data(entries: List[dict]) -> dict:
    """拼接 speaker 内容，提取分析，生成 segment 并统计时间"""
    full_text = " ".join(e["text"] for e in entries)
    analysis = extract_key_elements(full_text)

    # Segment 列表
    segments = []
    for idx, e in enumerate(entries):
        segments.append({
            "id": f"section{idx + 1}",
            "start": e.get("start", 0),
            "end": e.get("end", 0),
            "text": e.get("text", "")
        })

    # 起止时间
    starts = [e.get("start", 0) for e in entries if "start" in e]
    ends = [e.get("end", 0) for e in entries if "end" in e]
    start_time = min(starts) if starts else 0
    end_time = max(ends) if ends else 0

    return {
        **analysis,
        "segments": segments,
        "start": start_time,
        "end": end_time
    }

def generate_summary(speaker_analysis: Dict[str, dict], speaker_segments: Dict[str, List[dict]]) -> List[dict]:
    """为每个 speaker 生成摘要，同时包含该 speaker 整段说话的开始/结束时间与 section ID。"""
    summary_list = []

    for i, (speaker, analysis) in enumerate(speaker_analysis.items(), 1):
        # 清洗过长/过短条目
        def clean(lst: List[str]) -> List[str]:
            return [s for s in lst if 6 <= len(s) <= 200]

        actions = clean(analysis["actions"])
        decisions = clean(analysis["decisions"])
        conflicts = clean(analysis["conflicts"])

        ka = actions[:2]
        kd = decisions[:1]
        kc = conflicts[:1]

        parts = []

        # Action part
        if ka and ka[0] != "No specific actions identified":
            doc = nlp(ka[0])
            root = next((t.lemma_ for t in doc if t.dep_ == "ROOT"), None)
            if root:
                tail = ka[0][len(root):].strip()
                more = f" (and {len(actions)-1} more)" if len(actions) > 1 else ""
                parts.append(f"{root} {tail}{more}")
            else:
                parts.append(f"proposed action: {ka[0]}")
        else:
            parts.append("proposed no specific actions")

        # Decision part
        if kd and kd[0] != "No explicit decisions found":
            d0 = kd[0]
            if len(d0) > 80:
                d0 = d0[:80] + "..."
            parts.append(f"decided: \"{d0}\"")
        else:
            parts.append("made no major decisions")

        # Conflict part
        if kc and kc[0] != "No conflicts mentioned":
            c0 = kc[0]
            if len(c0) > 80:
                c0 = c0[:80] + "..."
            parts.append(f"raised concern: \"{c0}\"")
        else:
            parts.append("had no significant conflicts")

        summary_text = speaker + " " + "; ".join(parts) + "."

        # 获取时间信息
        segments = speaker_segments.get(speaker, [])
        if segments:
            start_time = min(seg["start"] for seg in segments)
            end_time = max(seg["end"] for seg in segments)
        else:
            start_time = end_time = None

        summary_list.append({
            "id": f"section{i}",
            "speaker": speaker,
            "start": round(start_time, 3) if start_time is not None else None,
            "end": round(end_time, 3) if end_time is not None else None,
            "actions": analysis["actions"],
            "decisions": analysis["decisions"],
            "conflicts": analysis["conflicts"],
            "summary": summary_text
        })

    return summary_list


def main(input_path: str, output_path: str):
    # 读取 diarized JSON
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 分组 by speaker
    by_speaker = defaultdict(list)
    for rec in records:
        spk = rec.get("speaker", "Unknown")
        text = rec.get("text", "").strip()
        if text:
            by_speaker[spk].append(rec)

    # 提取
    analysis = {spk: process_speaker_data(v) for spk, v in by_speaker.items()}

    # 生成 summary
    summary = generate_summary(analysis, by_speaker)

    # 写入
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed. Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Meeting diarized JSON → English summary with metadata")
    parser.add_argument(
        "--input",
        default="dataset/ICSI/diarized_json/Bed002_diarized.json",
        help="Path to diarized JSON"
    )
    parser.add_argument(
        "--output",
        default="dataset/ICSI/after_json/Bed002_summary.json",
        help="Path to output summary JSON"
    )
    args = parser.parse_args()
    main(args.input, args.output)

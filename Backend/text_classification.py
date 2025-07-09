#!/usr/bin/env python3
# text_classification.py

import json
import argparse
import datetime
from collections import defaultdict
import spacy

# —— 配置 & 工具函数 —— #

# 加载 spaCy 英文模型
nlp = spacy.load("en_core_web_md")

def iso_to_seconds(ts: str, base: datetime.datetime) -> float:
    """
    把 ISO 时间戳转换成秒，基于分会的 startTime 计算相对秒数
    """
    t = datetime.datetime.fromisoformat(ts)
    return (t - base).total_seconds()

def extract_key_elements(text: str) -> dict:
    """
    基于 spaCy + 关键词，提取 actions / decisions / conflicts
    返回原始句列表，和一句话 summary
    """
    doc = nlp(text)
    actions, decisions, conflicts = [], [], []

    decision_kw = {"decide","agree","choose","conclude","select","option"}
    conflict_kw = {"but","however","although","issue","problem","concern","disagree"}

    for sent in doc.sents:
        s = sent.text.strip()
        low = s.lower()
        # 决策
        if any(k in low for k in decision_kw):
            decisions.append(s)
        # 冲突
        if any(k in low for k in conflict_kw):
            conflicts.append(s)
        # 行动：根动词 + 宾语
        root = [t for t in sent if t.dep_=="ROOT"]
        if root and any(ch.dep_ in ("dobj","xcomp","ccomp") for ch in root[0].children):
            actions.append(s)

    # 最少保证不空
    if not actions:    actions = ["No actions detected"]
    if not decisions:  decisions = ["No decisions detected"]
    if not conflicts:  conflicts = ["No conflicts detected"]

    # 构造一句话 summary
    summary = (
        f"Actions: {actions[0]}{'...' if len(actions)>1 else ''}; "
        f"Decisions: {decisions[0]}{'...' if len(decisions)>1 else ''}; "
        f"Conflicts: {conflicts[0]}{'...' if len(conflicts)>1 else ''}."
    )
    return {
        "actions":    actions,
        "decisions":  decisions,
        "conflicts":  conflicts,
        "summary":    summary
    }

# —— 主流程 —— #

def main(api_json_path, aligned_json_path, output_path):
    # 1. 载入 API JSON
    with open(api_json_path, "r", encoding="utf-8") as f:
        api = json.load(f)

    # 必须有 startTime & agenda 列表
    meeting_start = api.get("startTime")
    agenda = api.get("agenda")
    if not meeting_start or not isinstance(agenda, list):
        print("⚠️ 未从 API JSON 中读取到有效的 startTime/agenda")
        return
    base_dt = datetime.datetime.fromisoformat(meeting_start)

    # 2. 构造 sections：计算每个条目的相对开始/结束秒数
    sections = []
    for item in agenda:
        title = item.get("title","")
        num   = item.get("number","")
        cs    = item.get("calculatedStartTime")
        length = item.get("lengthMinutes", 0)
        if not cs: continue
        start_s = iso_to_seconds(cs, base_dt)
        end_s   = start_s + length*60
        sections.append({
            "number": num,
            "title":   title,
            "start":   start_s,
            "end":     end_s
        })

    # 3. 读对齐 JSON，把每段 text 按 start 分配到对应 section
    with open(aligned_json_path, "r", encoding="utf-8") as f:
        aligned = json.load(f)

    by_section = defaultdict(list)
    for seg in aligned:
        t0 = seg.get("start",0)
        # 找到第一个满足 t0 in [sec.start, sec.end) 的 section
        for sec in sections:
            if sec["start"] <= t0 < sec["end"]:
                by_section[ (sec["number"],sec["title"]) ].append(seg["text"])
                break

    # 4. 对每个 section 下的文本做提取 & 总结
    output = []
    for num,title in by_section:
        texts = by_section[(num,title)]
        full_text = " ".join(texts)
        analysis = extract_key_elements(full_text)

        output.append({
            "section_number": num,
            "section_title":  title,
            "analysis":       analysis,
            "raw_texts":      texts
        })

    # 5. 写文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meeting_id":   api.get("meeting_id"),
            "meeting_name": api.get("name", ""),
            "sections":     output
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 结果已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基于 API 分段 + 对齐结果，做 per-section 分类 & 摘要"
    )
    parser.add_argument("api_json",    help="从后端拿到的 all_meetings JSON")
    parser.add_argument("aligned_json",help="diarize_and_align 脚本生成的 aligned.json")
    parser.add_argument("output",      help="输出 JSON 文件路径")
    args = parser.parse_args()

    main(args.api_json, args.aligned_json, args.output)
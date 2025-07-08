#!/usr/bin/env python3
# text_classification.py

import json
import torch
from pathlib import Path
from transformers import pipeline

# ── 配置：把你的输入/输出文件名写在这里 ───────────────────────────────────────
# 输入文件应该是 combined_api 脚本生成、按 agenda 切分并带有 "lines" 的 JSON
INPUT_JSON = "segmented_meeting_data.json"
# 自动在同目录、同文件名基础上加前缀
INPUT_PATH  = Path(INPUT_JSON)
OUTPUT_JSON = str(INPUT_PATH.parent / f"classified_{INPUT_PATH.name}")
# ────────────────────────────────────────────────────────────────────────────────

def load_data(path: str):
    """Load JSON from path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, path: str):
    """Save JSON to path."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def classify_sections(input_path: str, output_path: str):
    """
    对每个 agenda section 下的所有 lines 合并文本后做 zero-shot 分类，
    并把 label 和 label_score 写回到每个 agenda item 中。
    """
    # 1. 准备 classifier（zero-shot）
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        batch_size=8
    )

    # 2. 候选标签列表
    candidate_labels = ["action", "decision", "conflict", "other"]

    # 3. 加载整份 JSON（支持单会话或多会话格式）
    data = load_data(input_path)
    if isinstance(data, dict) and "meetings" in data:
        meetings = data["meetings"]
        wrap_key = "meetings"
    else:
        meetings = [data]
        wrap_key = None

    # 4. 遍历所有会议和每个 agenda item
    for meeting in meetings:
        for item in meeting.get("agenda", []):
            # 合并本节所有 utterance/text
            lines = item.get("lines", [])
            text = " ".join([ln.get("text", "").strip() for ln in lines]).strip()

            # 如果没有文本则跳过
            if not text:
                item["label"] = None
                item["label_score"] = None
                continue

            # 做分类
            result = classifier(text, candidate_labels=candidate_labels)
            item["label"]       = result["labels"][0]
            item["label_score"] = float(result["scores"][0])

    # 5. 保存结果，保持原结构
    if wrap_key:
        out_data = {wrap_key: meetings}
    else:
        out_data = meetings[0]

    save_data(out_data, output_path)
    print(f"✅ Classification done. Saved to {output_path}")

if __name__ == "__main__":
    classify_sections(INPUT_JSON, OUTPUT_JSON)

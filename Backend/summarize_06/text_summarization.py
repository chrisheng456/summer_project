#!/usr/bin/env python3
# text_summarization.py

import os
import json
import torch
import re
from pathlib import Path
from transformers import pipeline, AutoTokenizer

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
INPUT_JSON  = "classified_segmented_meeting_data.json"
INPUT_PATH  = Path(INPUT_JSON)
OUTPUT_JSON = str(INPUT_PATH.parent / f"summarized_{INPUT_PATH.name}")
MODEL_NAME  = "sshleifer/distilbart-cnn-12-6"
MAX_LENGTH  = 80
MIN_LENGTH  = 20
# ────────────────────────────────────────────────────────────────────────────────

def load_data(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_into_chunks(text: str, tokenizer, max_tokens: int):
    """
    根据 tokenizer.model_max_length，把 text 拆成多个不超过 max_tokens 的子块。
    这里先按句子拆，再累加句子直到接近 max_tokens。
    """
    # 简单的句子分隔
    sentences = re.split(r'(?<=[。？！\.!?])\s*', text)
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        if not sent:
            continue
        toks = tokenizer.encode(sent, add_special_tokens=False)
        if current_len + len(toks) > max_tokens:
            if current:
                chunks.append("".join(current))
            current = [sent]
            current_len = len(toks)
        else:
            current.append(sent)
            current_len += len(toks)
    if current:
        chunks.append("".join(current))
    return chunks

def main():
    data = load_data(str(INPUT_PATH))
    if isinstance(data, dict) and "meetings" in data:
        meetings = data["meetings"]
        wrap_key = "meetings"
    else:
        meetings = [data]
        wrap_key = None

    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=MODEL_NAME,
        device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_input_tokens = tokenizer.model_max_length  # typically 1024

    total_sections = 0
    for meeting in meetings:
        for item in meeting.get("agenda", []):
            total_sections += 1
            lines = item.get("lines", [])
            text  = " ".join([ln.get("text", "").strip() for ln in lines]).strip()
            if not text:
                item["summary"] = ""
                continue

            # 如果文本过长，先拆分
            chunks = split_into_chunks(text, tokenizer, max_input_tokens - 50)
            summaries = []
            for chunk in chunks:
                out = summarizer(
                    chunk,
                    max_length=MAX_LENGTH,
                    min_length=MIN_LENGTH,
                    do_sample=False
                )
                summaries.append(out[0]["summary_text"])
            # 合并所有 chunk 的摘要，再做一次整体压缩（可选）
            joined = " ".join(summaries)
            if len(tokenizer.encode(joined, add_special_tokens=False)) > max_input_tokens:
                # 如果合并后还是太长，再次摘要
                out = summarizer(
                    joined,
                    max_length=MAX_LENGTH,
                    min_length=MIN_LENGTH,
                    do_sample=False
                )
                item["summary"] = out[0]["summary_text"]
            else:
                item["summary"] = joined

    if wrap_key:
        out_data = {wrap_key: meetings}
    else:
        out_data = meetings[0]

    save_data(out_data, OUTPUT_JSON)
    print(f"✅ Summarized {total_sections} sections → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

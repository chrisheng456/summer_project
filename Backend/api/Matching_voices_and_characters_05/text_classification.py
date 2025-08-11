#!/usr/bin/env python3
# text_classification.py

import json
import re
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
INPUT_JSON  = "segmented_meeting_data.json"
INPUT_PATH  = Path(INPUT_JSON)
OUTPUT_JSON = str(INPUT_PATH.parent / f"classified_{INPUT_PATH.name}")

# summarization model for explanations
EXPL_MODEL     = "sshleifer/distilbart-cnn-12-6"
MAX_EXPL_TOKS  = 25
MIN_EXPL_TOKS  = 5
# ────────────────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_chunks(text, tokenizer, max_tokens):
    """Split text into chunks of <= max_tokens on sentence boundaries."""
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, current, cur_len = [], [], 0
    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        if cur_len + len(toks) > max_tokens:
            chunks.append(" ".join(current))
            current, cur_len = [], 0
        current.append(sent)
        cur_len += len(toks)
    if current:
        chunks.append(" ".join(current))
    return chunks

def classify_sections(input_path: str, output_path: str):
    # device setup
    device = 0 if torch.cuda.is_available() else -1

    # zero‐shot classifier
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        batch_size=8
    )

    # summarizer + tokenizer for explanations
    summarizer = pipeline(
        "summarization",
        model=EXPL_MODEL,
        device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(EXPL_MODEL)
    max_model_tokens = tokenizer.model_max_length

    labels = ["action", "decision", "conflict", "other"]
    data   = load_data(input_path)

    # support single‐meeting or batch under "meetings"
    if isinstance(data, dict) and "meetings" in data:
        meetings, wrap_key = data["meetings"], "meetings"
    else:
        meetings, wrap_key = [data], None

    for meeting in meetings:
        for item in meeting.get("agenda", []):
            texts     = [ln.get("text","").strip() for ln in item.get("lines",[])]
            full_text = " ".join(texts).strip()

            if not full_text:
                item["label"]        = None
                item["label_score"]  = None
                item["explanation"]  = ""
                continue

            # 1) zero‐shot classification
            res = classifier(full_text, candidate_labels=labels)
            label = res["labels"][0]
            score = float(res["scores"][0])
            item["label"], item["label_score"] = label, score

            # 2) for action/decision/conflict, generate one‐sentence explanation
            if label in {"action", "decision", "conflict"}:
                # choose prompt based on label
                if label == "action":
                    prompt_prefix = "Summarize the required action in one sentence: "
                elif label == "decision":
                    prompt_prefix = "Summarize the decision made in one sentence: "
                else:  # conflict
                    prompt_prefix = "Summarize the conflict of interest disclosed in one sentence: "

                # split if too long
                if len(tokenizer.encode(full_text, add_special_tokens=False)) > max_model_tokens - 50:
                    chunks = split_chunks(full_text, tokenizer, max_model_tokens - 50)
                else:
                    chunks = [full_text]

                exps = []
                for ch in chunks:
                    out = summarizer(
                        prompt_prefix + ch,
                        max_length=MAX_EXPL_TOKS,
                        min_length=MIN_EXPL_TOKS,
                        do_sample=False
                    )
                    exps.append(out[0]["summary_text"].strip())

                explanation = " ".join(exps).strip()
                # final squeeze if still too long
                if len(tokenizer.encode(explanation, add_special_tokens=False)) > MAX_EXPL_TOKS:
                    out2 = summarizer(
                        explanation,
                        max_length=MAX_EXPL_TOKS,
                        min_length=MIN_EXPL_TOKS,
                        do_sample=False
                    )
                    explanation = out2[0]["summary_text"].strip()

                item["explanation"] = explanation
            else:
                item["explanation"] = ""

    # write back
    out_data = {wrap_key: meetings} if wrap_key else meetings[0]
    save_data(out_data, output_path)
    print(f"✅ Classification and explanation done. Saved to {output_path}")

if __name__ == "__main__":
    classify_sections(str(INPUT_PATH), OUTPUT_JSON)

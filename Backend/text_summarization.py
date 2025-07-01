#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load a classified JSON, run abstractive summarization on each utterance’s text,
and write out a new JSON with "_summarized" suffix.

Assumes input records look like:
[
  {
    "speaker": "SPEAKER_01",
    "text": "…",
    "label": "action_item",
    "label_score": 0.83
  },
  …
]
After running, each record will also have a "summary" field.
"""

import os
import json
import torch
from transformers import pipeline

# ─── CONFIG ───────────────────────────────────────────
# path to your classified JSON:
INPUT_PATH = "dataset/ICSI/classified_json/Bdb001_classified.json"
# model choice:
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# summary length control:
MAX_LENGTH = 80
MIN_LENGTH = 20
# ──────────────────────────────────────────────────────

def main():
    # derive output path by appending "_summarized" before the .json
    base, ext = os.path.splitext(INPUT_PATH)
    OUTPUT_PATH = f"{base}_summarized{ext}"

    # load data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # init summarizer
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=MODEL_NAME,
        device=device,
    )

    # process each record
    for rec in records:
        text = rec.get("text", "").strip()
        if not text:
            rec["summary"] = ""
            continue

        # run summarization
        out = summarizer(
            text,
            max_length=MAX_LENGTH,
            min_length=MIN_LENGTH,
            do_sample=False,
        )
        rec["summary"] = out[0]["summary_text"]

    # save new JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(records)} utterances summarized → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

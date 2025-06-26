#!/usr/bin/env python3
# extract_meeting_insights_semantic.py

import json
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

CANDIDATE_LABELS = [
    "agenda item",
    "decision",
    "action item",
    "conflict of interest"
]
LABEL_KEY_MAP = {
    "agenda item": "agenda",
    "decision": "decisions",
    "action item": "actions",
    "conflict of interest": "conflicts"
}
CLASSIFIER_MODEL = "facebook/bart-large-mnli"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def semantic_label(text, classifier):
    out = classifier(text, CANDIDATE_LABELS)
    label, score = out["labels"][0], out["scores"][0]
    return label if score >= 0.5 else None

def extract_by_label(texts, key, extractor):
    if not texts:
        return []
    prompt = (
        f"You are extracting {key} from meeting transcripts. Return only a JSON list of objects.\n"
        + "\n".join(texts)
    )
    resp = extractor(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    start = resp.find("[")
    end   = resp.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return []
    fragment = resp[start:end+1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diarized_json")
    parser.add_argument("output_json")
    parser.add_argument("extract_model")
    parser.add_argument("--chunk_size", type=int, default=8)
    args = parser.parse_args()

    device_index = 0 if torch.cuda.is_available() else -1
    print(f"→ device: {'cuda' if device_index==0 else 'cpu'}")

    # zero-shot classifier on device
    classifier = pipeline(
        "zero-shot-classification",
        model=CLASSIFIER_MODEL,
        device=device_index
    )

    diarized = load_json(args.diarized_json)

    buckets = {v: [] for v in LABEL_KEY_MAP.values()}
    for seg in diarized:
        text = seg.get("text", "")
        label = semantic_label(text, classifier)
        if label:
            key = LABEL_KEY_MAP[label]
            buckets[key].append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker']}: {text}")

    # load extract model with accelerate-compatible device map
    use_gpu = (device_index == 0)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.extract_model,
        device_map="auto" if use_gpu else {"": "cpu"},
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)

    # pipeline: no device arg when using accelerate
    if use_gpu:
        extractor = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
    else:
        extractor = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )

    insights = {}
    for key, texts in buckets.items():
        print(f"→ Extracting {key} ({len(texts)} items)")
        insights[key] = extract_by_label(texts, key, extractor)

    save_json(insights, args.output_json)
    print(f"→ Saved insights to {args.output_json}")

if __name__ == "__main__":
    main()

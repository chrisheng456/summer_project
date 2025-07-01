#!/usr/bin/env python3
# zero_shot_classification.py

import json
import torch
from transformers import pipeline

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def classify_utterances(input_path: str, output_path: str):
    # 1. 准备分类器：zero-shot pipeline
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        batch_size=8
    )

    # 2. 定义候选标签
    candidate_labels = ["action", "decision", "conflict", "other"]

    # 3. 加载数据
    data = load_data(input_path)

    # 4. 遍历每条 utterance，做分类
    for utt in data:
        text = utt.get("text", "").strip()
        if not text:
            utt["label"] = None
            utt["label_score"] = None
            continue

        # zero-shot 返回 labels 和 scores 列表
        result = classifier(text, candidate_labels=candidate_labels)
        utt["label"] = result["labels"][0]
        utt["label_score"] = float(result["scores"][0])

    # 5. 保存带分类结果的数据
    save_data(data, output_path)
    print(f"Classification done. Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zero-shot classify meeting utterances")
    parser.add_argument("--input",  "-i", required=True,  help="Path to input JSON (list of {{'text':...}})")
    parser.add_argument("--output", "-o", required=True,  help="Path to output JSON")
    args = parser.parse_args()
    classify_utterances(args.input, args.output)

import os
import re
import json
import spacy
from collections import defaultdict
from typing import List, Dict

# 加载 spaCy 英文模型（首次使用前需执行：python -m spacy download en_core_web_md）
nlp = spacy.load("en_core_web_md")

def extract_key_elements(text: str) -> dict:
    """使用 spaCy 依存分析提取 actions/decisions/conflicts，事先过滤噪声句子。"""
    doc = nlp(text)

    # 过滤噪声句子：极短的填充词或首字母重复
    def is_noise_sentence(t: str) -> bool:
        words = t.strip().split()
        if len(words) < 4 and re.fullmatch(r'([A-Za-z]\s*){1,4}', t):
            return True
        return False

    filtered_sents = [sent for sent in doc.sents if not is_noise_sentence(sent.text)]

    # 1. Actions: 根动词（MD/VB）且有直接宾语
    actions = []
    for sent in filtered_sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.tag_ in ("MD", "VB"):
                if any(ch.dep_ == "dobj" for ch in token.children):
                    phrase = " ".join(t.text for t in sent if not t.is_punct)
                    actions.append(phrase)

    # 2. Decisions: 包含决策关键词
    decisions = []
    decision_keywords = ["decide", "choose", "select", "option", "prefer", "go with", "agree", "conclude"]
    for sent in filtered_sents:
        low = sent.text.lower()
        if any(k in low for k in decision_keywords):
            phrase = " ".join(t.text for t in sent if not t.is_punct)
            decisions.append(phrase)

    # 3. Conflicts: 包含冲突关键词
    conflicts = []
    conflict_keywords = ["but", "however", "although", "though", "disagree", "problem", "issue", "concern", "doubt"]
    for sent in filtered_sents:
        low = sent.text.lower()
        if any(k in low for k in conflict_keywords):
            phrase = " ".join(t.text for t in sent if not t.is_punct)
            conflicts.append(phrase)

    return {
        "actions":   actions   or ["No specific actions identified"],
        "decisions": decisions or ["No explicit decisions found"],
        "conflicts": conflicts or ["No conflicts mentioned"],
    }

def process_speaker_data(entries: List[dict]) -> dict:
    """将同一 speaker 的所有 text 拼接后提取关键元素。"""
    full_text = " ".join(e["text"] for e in entries)
    return extract_key_elements(full_text)

def generate_summary(speaker_analysis: Dict[str, dict]) -> List[dict]:
    """为每个 speaker 生成一句话式英文 summary，并保留完整列表。"""
    summary_list = []

    for speaker, analysis in speaker_analysis.items():
        # 清洗过长/过短条目
        def clean(lst: List[str]) -> List[str]:
            return [s for s in lst if 6 <= len(s) <= 200]

        actions   = clean(analysis["actions"])
        decisions = clean(analysis["decisions"])
        conflicts = clean(analysis["conflicts"])

        # 用于 summary 的首条
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
                more = f" (and {len(actions)-1} more)" if len(actions)>1 else ""
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

        summary_list.append({
            "speaker":  speaker,
            "actions":  analysis["actions"],
            "decisions":analysis["decisions"],
            "conflicts":analysis["conflicts"],
            "summary":  summary_text
        })

    return summary_list

def main(input_path: str, output_path: str):
    # 1. 读取 diarized JSON
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 2. 按 speaker 分组
    by_speaker = defaultdict(list)
    for rec in records:
        spk = rec.get("speaker", "Unknown")
        text = rec.get("text", "").strip()
        if text:
            by_speaker[spk].append(rec)

    # 3. 提取
    analysis = {spk: process_speaker_data(v) for spk, v in by_speaker.items()}

    # 4. 生成 summary
    summary = generate_summary(analysis)

    # 5. 输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed. Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Meeting diarized JSON → English summary")
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
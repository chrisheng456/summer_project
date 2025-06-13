"""
post_process_local.py
本地生成 meeting_minutes_xxx_local.json
"""

import json, re
from pathlib import Path
import yake, spacy
from transformers import pipeline

# 1. 找到最近的转写 JSON
fp = sorted(Path(".").glob("meeting_minutes_*.json"))[-1]
data = json.loads(fp.read_text(encoding="utf-8"))
transcript = data["transcription"]

# 2. 摘要
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",   # 已下载约 1 GB
    device=-1,                               # -1 = 纯 CPU
    model_kwargs={
        "low_cpu_mem_usage": False,          # 禁掉低内存模式
        "device_map": None                   # 不再尝试自动拆包/离线
    }
)
summary = summarizer(transcript, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

# 3. 关键词（YAKE）
kw = yake.KeywordExtractor(lan="en", n=2, top=6)
key_points = [k for k,_ in kw.extract_keywords(transcript)]

# 4. 行动项（简单规则）
nlp = spacy.load("en_core_web_sm")
action_items = []
for sent in nlp(transcript).sents:
    if re.match(r"(?i)(we|please|need to|let's)\b", sent.text.strip()):
        action_items.append({
            "task": sent.text.strip(),
            "owner": "Unknown",
            "due"  : None
        })

# 5. 情感（多语言 BERT 1–5 星）
sentiment_pipe = pipeline("sentiment-analysis",
                          model="nlptown/bert-base-multilingual-uncased-sentiment")
sentiment = sentiment_pipe(summary[:512])[0]["label"]   # e.g. "4 stars"

# 6. 写回
data.update({
    "abstract_summary": summary,
    "key_points": key_points,
    "action_items": action_items,
    "sentiment": sentiment,
})
out = fp.with_name(fp.stem + "_local.json")
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ Done → {out.name}")

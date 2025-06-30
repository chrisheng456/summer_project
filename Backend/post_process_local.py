import json
import logging
from pathlib import Path
from datetime import datetime
from dateparser.search import search_dates

import yake
import spacy
from transformers import pipeline
from spacy.matcher import Matcher

# 配置区
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
ZS_MODEL = "facebook/bart-large-mnli"
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_CHUNK_TOKENS = 1024
SUMMARY_MIN_LEN = 30
SUMMARY_MAX_LEN = 120
YAKE_LANG = "en"
YAKE_MAX_NGRAM = 2
LOG_LEVEL = logging.INFO

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def find_latest_file(pattern="meeting_minutes_*.json"):
    files = list(Path(".").glob(pattern))
    if not files:
        logging.error("未找到任何匹配的文件：%s", pattern)
        raise FileNotFoundError(f"No files matching {pattern}")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    logging.info("找到最新文件：%s", latest.name)
    return latest

def chunk_text(text, max_chars=8000):
    """简单按字符分块（近似控制在模型输入限制内）"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # 尽量按句号断开
        if end < len(text):
            end = text.rfind(".", start, end) + 1 or end
        chunks.append(text[start:end].strip())
        start = end
    logging.info("分成 %d 个块用于摘要", len(chunks))
    return chunks

def summarize_text(chunks, model, tokenizer, **kwargs):
    summaries = []
    for chunk in chunks:
        try:
            out = model(chunk, **kwargs)[0]["summary_text"]
            summaries.append(out.strip())
        except Exception as e:
            logging.warning("摘要模型块失败：%s", e)
    return " ".join(summaries)

def extract_keywords(text, lang=YAKE_LANG, ngram=YAKE_MAX_NGRAM):
    kw_extractor = yake.KeywordExtractor(lan=lang, n=ngram, top=1)
    # 根据长度动态决定 top_k
    top_k = min(10, max(5, len(text) // 1000))
    kw_extractor.top = top_k
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(text)]
    logging.info("提取关键词 %d 条", len(keywords))
    return keywords

def init_action_extractors(nlp):
    matcher = Matcher(nlp.vocab)
    # 添加一些常见动词模式
    matcher.add("ACTION_VERB", [
        [
            {"LOWER": {"IN": ["please", "kindly"]}, "OP": "?"},
            {"POS": "VERB"},
            {"POS": {"IN": ["DET", "ADV"]}, "OP": "*"},
            {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
        ]
    ])
    # 零样本分类管道
    zs_pipe = pipeline("zero-shot-classification", model=ZS_MODEL, device=-1)
    return matcher, zs_pipe

def is_action_sentence(sent, matcher, zs_pipe, threshold=0.7):
    doc = sent.as_doc()
    # 1. 依存根是动词原形（祈使句）
    if sent.root.pos_ == "VERB" and sent.root.tag_ in ("VB", "VBP"):
        return True
    # 2. Matcher 模式
    if matcher(doc):
        return True
    # 3. 零样本分类
    try:
        res = zs_pipe(sent.text, candidate_labels=["action item", "not action"])
        score = res["scores"][res["labels"].index("action item")]
        return score >= threshold
    except Exception as e:
        logging.warning("零样本分类失败：%s", e)
        return False

def extract_owner_and_due(text, nlp):
    doc = nlp(text)
    # 负责人：取第一 PERSON 实体
    owners = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    owner = owners[0] if owners else None
    # 截止日期：用 dateparser 搜索
    dates = search_dates(text, languages=["en"])
    due = dates[0][1].date().isoformat() if dates else None
    return owner, due

def main():
    setup_logging()
    try:
        fp = find_latest_file()
        data = json.loads(fp.read_text(encoding="utf-8"))
        transcript = data.get("transcription", "")
        if not transcript:
            logging.error("transcription 字段为空")
            return

        # 1. 分块摘要
        # 1. 分块摘要（改用 XSum，直接生成更短的摘要）
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-xsum",
            device=-1
        )
        # 如果 transcript 很长，也可以先分块拼接再调用；不过 XSum 模型一般能接受较短的整段
        # 这里我们假设 transcript 不会超出模型限制：
        summary = summarizer(
            transcript,
            max_length=60,
            min_length=15,
            do_sample=False
        )[0]["summary_text"]

        # 2. 关键词
        key_points = extract_keywords(transcript)

        # 3. 行动项
        nlp = spacy.load("en_core_web_sm")
        matcher, zs_pipe = init_action_extractors(nlp)

        action_items = []
        for sent in nlp(transcript).sents:
            if is_action_sentence(sent, matcher, zs_pipe):
                # 正确地传入 nlp 对象
                owner, due = extract_owner_and_due(sent.text, nlp)
                action_items.append({
                    "task": sent.text.strip(),
                    "owner": owner or "TBD",
                    "due": due
                })
        logging.info("共提取 %d 条行动项", len(action_items))

        # 4. 情感
        sentiment_pipe = pipeline("sentiment-analysis",
                                  model=SENTIMENT_MODEL,
                                  device=-1)
        sent_res = sentiment_pipe(summary[:512])[0]["label"]
        # 把 “4 stars” 映射到数值
        sentiment_score = int(sent_res.split()[0]) if sent_res[0].isdigit() else None

        # 写回
        data.update({
            "abstract_summary": summary,
            "key_points": key_points,
            "action_items": action_items,
            "sentiment": {
                "label": sent_res,
                "score": sentiment_score
            },
            "processed_at": datetime.utcnow().isoformat() + "Z"
        })
        out = fp.with_name(fp.stem + "_local.json")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("✅ Done → %s", out.name)

    except Exception as e:
        logging.exception("处理失败：%s", e)

if __name__ == "__main__":
    main()
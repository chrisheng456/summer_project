# Backend/api/app/pipeline/s05_text_classification/__init__.py
from __future__ import annotations

import re
from typing import List, Dict
import torch
from loguru import logger
from transformers import pipeline, AutoTokenizer

from ...schema.process_information import ProcessInformation

# —— 摘要模型（用于“一句话解释”） ——
EXPL_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_EXPL_TOKS = 100
MIN_EXPL_TOKS = 5

# —— 分类候选标签 ——
LABELS = ["action", "decision", "conflict", "other"]

# —— 批量大小（按显存调优） ——
BATCH_SIZE_CLASSIFY = 8
BATCH_SIZE_SUMMARY = 8


class TextClassificationPipeline:
    """
    改进版：
    - 对“分类文本”按 token 上限切块，使用 zero-shot 批量推理并聚合得分
    - 解释（summarizer）也按上限切块，批量生成后再压缩
    - 复用 pipeline 实例，避免重复初始化
    """

    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1

        # —— Zero-shot 分类器 + 分词器（BART MNLI，最大 1024 tokens）——
        self.clf_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.clf_max_len = self.clf_tokenizer.model_max_length  # 通常 1024
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            tokenizer=self.clf_tokenizer,
            device=device,
            batch_size=BATCH_SIZE_CLASSIFY,
            # 安全兜底：即使误传长文本也会被截断；核心仍是我们自己的分块逻辑
            tokenizer_kwargs={"truncation": True, "max_length": self.clf_max_len},
        )

        # —— 摘要器 + 分词器（用于“一句话解释”）——
        self.sum_tokenizer = AutoTokenizer.from_pretrained(EXPL_MODEL)
        self.sum_max_len = self.sum_tokenizer.model_max_length  # 典型 1024
        self.summarizer = pipeline(
            "summarization",
            model=EXPL_MODEL,
            device=device,
            batch_size=BATCH_SIZE_SUMMARY,
        )

    # --------- 工具：按 token 上限切块（尽量在句子边界处断开） ---------
    def _split_by_tokens(self, text: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
        if not text:
            return []
        # 留出少量余量，避免特殊符号溢出
        budget = max(32, max_tokens - 50)
        sentences = re.split(r'(?<=[\.\!\?。！？])\s+', text)  # 英文/中英文句号兼容
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for sent in sentences:
            if not sent:
                continue
            ids = tokenizer.encode(sent, add_special_tokens=False)
            n = len(ids)
            if cur_len + n > budget:
                if cur:
                    chunks.append(" ".join(cur).strip())
                cur, cur_len = [sent], n
            else:
                cur.append(sent)
                cur_len += n
        if cur:
            chunks.append(" ".join(cur).strip())
        return [c for c in chunks if c]

    # --------- 工具：把 zero-shot 的多块结果做“分数聚合” ---------
    def _aggregate_zeroshot(self, results) -> Dict[str, float]:
        # results 既可能是单个 dict，也可能是 list[dict]（批量）
        if isinstance(results, dict):
            results = [results]
        agg = {lab: 0.0 for lab in LABELS}
        for res in results:
            for lab, sc in zip(res["labels"], res["scores"]):
                agg[lab] += float(sc)
        # 归一为“平均分”，便于不同块数量可比
        n = max(1, len(results))
        for k in agg:
            agg[k] /= n
        return agg

    # --------- 工具：批量摘要并拼接，然后再做一次压缩 ---------
    def _summarize_explanation(self, prompt_prefix: str, text: str) -> str:
        if not text:
            return ""

        # 1) 先按摘要器上限切块
        chunks = self._split_by_tokens(text, self.sum_tokenizer, self.sum_max_len)
        if not chunks:
            chunks = [text]

        # 2) 批量调用 summarizer（更高效）
        batched_inputs = [prompt_prefix + c for c in chunks]
        outs = self.summarizer(
            batched_inputs,
            max_length=MAX_EXPL_TOKS,
            min_length=MIN_EXPL_TOKS,
            do_sample=False,
        )
        pieces = [o[0]["summary_text"].strip() if isinstance(o, list) else o["summary_text"].strip() for o in outs]
        explanation = " ".join(pieces).strip()

        # 3) 如果仍然过长，再压一次
        if len(self.sum_tokenizer.encode(explanation, add_special_tokens=False)) > MAX_EXPL_TOKS:
            o2 = self.summarizer(
                explanation,
                max_length=MAX_EXPL_TOKS,
                min_length=MIN_EXPL_TOKS,
                do_sample=False,
            )
            explanation = (o2[0]["summary_text"] if isinstance(o2, list) else o2["summary_text"]).strip()

        return explanation

    # --------- 主流程 ---------
    def process(self, info: ProcessInformation):
        agenda = getattr(info, "customer_meeting_detail", {}).get("agenda", [])
        if not agenda:
            logger.warning("S05 TextClassification: 没有找到议程信息")
            return

        for item in agenda:
            texts = [(ln.get("text") or "").strip() for ln in item.get("lines", [])]
            full_text = " ".join(t for t in texts if t).strip()

            if not full_text:
                item["label"] = None
                item["label_score"] = None
                item["explanation"] = ""
                continue

            # —— (A) 分类：按 BART MNLI 上限切块 + 批量 zero-shot + 分数聚合 ——
            clf_chunks = self._split_by_tokens(full_text, self.clf_tokenizer, self.clf_max_len)
            if not clf_chunks:
                clf_chunks = [full_text]

            # 批量推理（pipeline 支持 list 输入）
            res_list = self.classifier(clf_chunks, candidate_labels=LABELS)
            agg = self._aggregate_zeroshot(res_list)
            label = max(agg, key=agg.get)
            score = float(agg[label])

            item["label"], item["label_score"] = label, score

            # —— (B) 解释：仅对三大类生成一句话解释，依旧用分块 + 批量摘要 ——
            if label in {"action", "decision", "conflict"}:
                if label == "action":
                    prefix = "Summarize the required action in one sentence: "
                elif label == "decision":
                    prefix = "Summarize the decision made in one sentence: "
                else:
                    prefix = "Summarize the conflict of interest disclosed in one sentence: "

                item["explanation"] = self._summarize_explanation(prefix, full_text)
            else:
                item["explanation"] = ""

        logger.info("S05 TextClassification: 分类与解释已完成。")

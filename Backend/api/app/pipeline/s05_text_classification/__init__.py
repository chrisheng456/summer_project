from __future__ import annotations

import re
from typing import List, Dict
import torch
from loguru import logger
from transformers import pipeline, AutoTokenizer

from ...schema.process_information import ProcessInformation

EXPL_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_EXPL_TOKS = 100
MIN_EXPL_TOKS = 5

LABELS = ["action", "decision", "conflict", "other"]

BATCH_SIZE_CLASSIFY = 8
BATCH_SIZE_SUMMARY = 8


class TextClassificationPipeline:
    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1

        self.clf_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.clf_max_len = self.clf_tokenizer.model_max_length
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            tokenizer=self.clf_tokenizer,
            device=device,
            batch_size=BATCH_SIZE_CLASSIFY,
            tokenizer_kwargs={"truncation": True, "max_length": self.clf_max_len},
        )

        self.sum_tokenizer = AutoTokenizer.from_pretrained(EXPL_MODEL)
        self.sum_max_len = self.sum_tokenizer.model_max_length
        self.summarizer = pipeline(
            "summarization",
            model=EXPL_MODEL,
            device=device,
            batch_size=BATCH_SIZE_SUMMARY,
        )

    def _split_by_tokens(self, text: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
        if not text:
            return []
        budget = max(32, max_tokens - 50)
        sentences = re.split(r'(?<=[\.\!\?])\s+', text)
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

    def _aggregate_zeroshot(self, results) -> Dict[str, float]:
        if isinstance(results, dict):
            results = [results]
        agg = {lab: 0.0 for lab in LABELS}
        for res in results:
            for lab, sc in zip(res["labels"], res["scores"]):
                agg[lab] += float(sc)
        n = max(1, len(results))
        for k in agg:
            agg[k] /= n
        return agg

    def _summarize_explanation(self, prompt_prefix: str, text: str) -> str:
        if not text:
            return ""

        chunks = self._split_by_tokens(text, self.sum_tokenizer, self.sum_max_len)
        if not chunks:
            chunks = [text]

        batched_inputs = [prompt_prefix + c for c in chunks]
        outs = self.summarizer(
            batched_inputs,
            max_length=MAX_EXPL_TOKS,
            min_length=MIN_EXPL_TOKS,
            do_sample=False,
        )
        pieces = [o[0]["summary_text"].strip() if isinstance(o, list) else o["summary_text"].strip() for o in outs]
        explanation = " ".join(pieces).strip()

        if len(self.sum_tokenizer.encode(explanation, add_special_tokens=False)) > MAX_EXPL_TOKS:
            o2 = self.summarizer(
                explanation,
                max_length=MAX_EXPL_TOKS,
                min_length=MIN_EXPL_TOKS,
                do_sample=False,
            )
            explanation = (o2[0]["summary_text"] if isinstance(o2, list) else o2["summary_text"]).strip()

        return explanation

    def process(self, info: ProcessInformation):
        agenda = getattr(info, "customer_meeting_detail", {}).get("agenda", [])
        if not agenda:
            logger.warning("s05 TextClassification: No agenda information found")
            return

        for item in agenda:
            texts = [(ln.get("text") or "").strip() for ln in item.get("lines", [])]
            full_text = " ".join(t for t in texts if t).strip()

            if not full_text:
                item["label"] = None
                item["label_score"] = None
                item["explanation"] = ""
                continue

            clf_chunks = self._split_by_tokens(full_text, self.clf_tokenizer, self.clf_max_len)
            if not clf_chunks:
                clf_chunks = [full_text]

            res_list = self.classifier(clf_chunks, candidate_labels=LABELS)
            agg = self._aggregate_zeroshot(res_list)
            label = max(agg, key=agg.get)
            score = float(agg[label])

            item["label"], item["label_score"] = label, score

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

        logger.info("s05 TextClassification: Classification and explanation completed.")

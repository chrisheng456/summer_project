import re
import torch
from loguru import logger
from transformers import pipeline, AutoTokenizer

from ...schema.process_information import ProcessInformation

EXPL_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_EXPL_TOKS = 25
MIN_EXPL_TOKS = 5


class TextClassificationPipeline:
    def process(self, info: ProcessInformation):
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
            batch_size=8,
        )
        summarizer = pipeline("summarization", model=EXPL_MODEL, device=device)
        tokenizer = AutoTokenizer.from_pretrained(EXPL_MODEL)
        max_model_tokens = tokenizer.model_max_length
        labels = ["action", "decision", "conflict", "other"]
        agenda = getattr(info, "customer_meeting_detail", {}).get("agenda", [])
        if not agenda:
            logger.warning("没有找到议程信息")
            return
        for item in agenda:
            texts = [
                ln.get("text", "").strip() for ln in item.get("lines", [])
            ]
            full_text = " ".join(texts).strip()
            if not full_text:
                item["label"] = None
                item["label_score"] = None
                item["explanation"] = ""
                continue
            res = classifier(full_text, candidate_labels=labels)
            label = res["labels"][0]
            score = float(res["scores"][0])
            item["label"], item["label_score"] = label, score
            if label in {"action", "decision", "conflict"}:
                if label == "action":
                    prompt_prefix = (
                        "Summarize the required action in one sentence: "
                    )
                elif label == "decision":
                    prompt_prefix = (
                        "Summarize the decision made in one sentence: "
                    )
                else:
                    prompt_prefix = "Summarize the conflict of interest disclosed in one sentence: "
                if (
                    len(tokenizer.encode(full_text, add_special_tokens=False))
                    > max_model_tokens - 50
                ):
                    sentences = re.split(r"(?<=[\.\!?])\s+", full_text)
                    chunks, current, cur_len = [], [], 0
                    for sent in sentences:
                        toks = tokenizer.encode(sent, add_special_tokens=False)
                        if cur_len + len(toks) > max_model_tokens - 50:
                            chunks.append(" ".join(current))
                            current, cur_len = [], 0
                        current.append(sent)
                        cur_len += len(toks)
                    if current:
                        chunks.append(" ".join(current))
                else:
                    chunks = [full_text]
                exps = []
                for ch in chunks:
                    out = summarizer(
                        prompt_prefix + ch,
                        max_length=MAX_EXPL_TOKS,
                        min_length=MIN_EXPL_TOKS,
                        do_sample=False,
                    )
                    exps.append(out[0]["summary_text"].strip())
                explanation = " ".join(exps).strip()
                if (
                    len(
                        tokenizer.encode(explanation, add_special_tokens=False)
                    )
                    > MAX_EXPL_TOKS
                ):
                    out2 = summarizer(
                        explanation,
                        max_length=MAX_EXPL_TOKS,
                        min_length=MIN_EXPL_TOKS,
                        do_sample=False,
                    )
                    explanation = out2[0]["summary_text"].strip()
                item["explanation"] = explanation
            else:
                item["explanation"] = ""

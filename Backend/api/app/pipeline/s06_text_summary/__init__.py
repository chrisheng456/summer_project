import re
import torch
from loguru import logger
from transformers import pipeline, AutoTokenizer

from ...schema.process_information import ProcessInformation

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_LENGTH = 80
MIN_LENGTH = 20


class TextSummaryPipeline:
    def process(self, info: ProcessInformation):
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model=MODEL_NAME, device=device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        max_input_tokens = tokenizer.model_max_length
        agenda = getattr(info, "customer_meeting_detail", {}).get("agenda", [])
        if not agenda:
            logger.warning("No agenda information found")
            return
        for item in agenda:
            lines = item.get("lines", [])
            text = " ".join(
                [ln.get("text", "").strip() for ln in lines]
            ).strip()
            if not text:
                item["summary"] = ""
                continue
            sentences = re.split(r"(?<=[\.!?])\s*", text)
            chunks, current = [], []
            current_len = 0
            for sent in sentences:
                if not sent:
                    continue
                toks = tokenizer.encode(sent, add_special_tokens=False)
                if current_len + len(toks) > max_input_tokens - 50:
                    if current:
                        chunks.append("".join(current))
                    current = [sent]
                    current_len = len(toks)
                else:
                    current.append(sent)
                    current_len += len(toks)
            if current:
                chunks.append("".join(current))
            summaries = []
            for chunk in chunks:
                out = summarizer(
                    chunk,
                    max_length=MAX_LENGTH,
                    min_length=MIN_LENGTH,
                    do_sample=False,
                )
                summaries.append(out[0]["summary_text"])
            joined = " ".join(summaries)
            if (
                len(tokenizer.encode(joined, add_special_tokens=False))
                > max_input_tokens
            ):
                out = summarizer(
                    joined,
                    max_length=MAX_LENGTH,
                    min_length=MIN_LENGTH,
                    do_sample=False,
                )
                item["summary"] = out[0]["summary_text"]
            else:
                item["summary"] = joined
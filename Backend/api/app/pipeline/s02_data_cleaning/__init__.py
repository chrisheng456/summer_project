import re
import spacy
from nltk.tokenize import sent_tokenize

from ...schema.process_information import ProcessInformation

nlp = spacy.load("en_core_web_sm")

# 仅保留真正“口头填充”的冗余词（删除）；保留有含义的语气词
PURE_FILLER_WORDS = {
    "uh","um","ah","er","hm","hmm","eh","mhm","uh-huh","uh-uh","erm","ummm","uhm","mm","mmm","huh",
}

# 缩写替换
ABBREVIATIONS = {
    "btw":"by the way","w/":"with","w/o":"without","e.g.":"for example","i.e.":"that is","etc.":"and so on",
    "&":"and","+":"plus","re:":"regarding","asap":"as soon as possible","fyi":"for your information",
}

# 简单噪音模式
NOISE_PATTERNS = [
    r"\([^)]*\)", r"\[[^\]]*\]", r"\b(?:cough|laugh|laughter|applause|breath|sigh|noise|static)\b",
    r"\b(?:\w*[\*#]\w*)\b",
]

def clean_text(text: str) -> str:
    # 1) 替换缩写
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text, flags=re.IGNORECASE)
    # 2) 删除冗余口头词
    filler = r"\b(?:" + "|".join(map(re.escape, PURE_FILLER_WORDS)) + r")\b"
    text = re.sub(filler, "", text, flags=re.IGNORECASE)
    # 3) 去噪
    for p in NOISE_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    # 4) 规范标点与空格
    text = re.sub(r"([,.?!;:])\1+", r"\1", text)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"([,.?!;:])(\w)", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[-1] not in {".","?","!"}:
        text += "."
    return text

class DataCleaningPipeline:
    def process(self, info: ProcessInformation):
        """
        就地覆盖每条识别行的 text；不再产生 tokens/lemmas/processed 等冗余字段。
        仅当 info.transcription 存在时处理。
        """
        if not getattr(info, "transcription", None):
            return
        for ln in info.transcription:
            t = (ln.get("text") or "").strip()
            if t:
                ln["text"] = clean_text(t)
        # 若前面有地方读取 cleaned_transcription，可按需复制一份：
        info.cleaned_transcription = info.transcription

import re
import spacy
from nltk.tokenize import sent_tokenize

from ...schema.process_information import ProcessInformation

nlp = spacy.load("en_core_web_sm")


PURE_FILLER_WORDS = {
    "uh","um","ah","er","hm","hmm","eh","mhm","uh-huh","uh-uh","erm","ummm","uhm","mm","mmm","huh",
}

ABBREVIATIONS = {
    "btw":"by the way","w/":"with","w/o":"without","e.g.":"for example","i.e.":"that is","etc.":"and so on",
    "&":"and","+":"plus","re:":"regarding","asap":"as soon as possible","fyi":"for your information",
}

NOISE_PATTERNS = [
    r"\([^)]*\)", r"\[[^\]]*\]", r"\b(?:cough|laugh|laughter|applause|breath|sigh|noise|static)\b",
    r"\b(?:\w*[\*#]\w*)\b",
]

def clean_text(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text, flags=re.IGNORECASE)
    filler = r"\b(?:" + "|".join(map(re.escape, PURE_FILLER_WORDS)) + r")\b"
    text = re.sub(filler, "", text, flags=re.IGNORECASE)
    for p in NOISE_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"([,.?!;:])\1+", r"\1", text)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"([,.?!;:])(\w)", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[-1] not in {".","?","!"}:
        text += "."
    return text

class DataCleaningPipeline:
    def process(self, info: ProcessInformation):
        if not getattr(info, "transcription", None):
            return
        for ln in info.transcription:
            t = (ln.get("text") or "").strip()
            if t:
                ln["text"] = clean_text(t)
        info.cleaned_transcription = info.transcription

import re
import spacy
from nltk.tokenize import sent_tokenize

from app.schema.process_information import ProcessInformation

nlp = spacy.load("en_core_web_sm")

PURE_FILLER_WORDS = {
    "uh",
    "um",
    "ah",
    "er",
    "hm",
    "hmm",
    "eh",
    "mhm",
    "uh-huh",
    "uh-uh",
    "erm",
    "ummm",
    "uhm",
    "mm",
    "mmm",
    "huh",
}

MEANINGFUL_INTERJECTIONS = {
    "yes",
    "no",
    "yeah",
    "yep",
    "nope",
    "okay",
    "ok",
    "right",
    "oh",
    "ah",
    "wow",
    "great",
    "good",
    "fine",
    "exactly",
    "absolutely",
    "certainly",
    "sure",
    "agreed",
    "understood",
    "got it",
    "please",
    "thanks",
    "thank you",
}

ABBREVIATIONS = {
    "btw": "by the way",
    "w/": "with",
    "w/o": "without",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "and so on",
    "vs.": "versus",
    "approx.": "approximately",
    "&": "and",
    "+": "plus",
    "re:": "regarding",
    "asap": "as soon as possible",
    "afaik": "as far as i know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "fyi": "for your information",
    "tbh": "to be honest",
    "n/a": "not applicable",
}

NOISE_PATTERNS = [
    r"\([^)]*\)",  # 删除括号内容
    r"\[[^\]]*\]",  # 删除方括号内容
    # 删除特定噪音词
    r"\b(?:cough|laugh|laughter|applause|breath|sigh|noise|static)\b",
    r"\b(?:\w*[\*#]\w*)\b",  # 删除含特殊符号的词
]


def clean_text(text):
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(
            rf"\b{re.escape(abbr)}\b", full, text, flags=re.IGNORECASE
        )
    filler_pattern = (
        r"\b(?:" + "|".join(map(re.escape, PURE_FILLER_WORDS)) + r")\b"
    )
    text = re.sub(filler_pattern, "", text, flags=re.IGNORECASE)
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"([,.?!;:])\1+", r"\1", text)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"([,.?!;:])(\w)", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    if text and text[-1] not in {".", "?", "!"}:
        text += "."
    return text


def should_keep_sentence(sentence):
    if not sentence.strip():
        return False
    doc = nlp(sentence)
    has_content = any(
        token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        or token.text.lower() in MEANINGFUL_INTERJECTIONS
        for token in doc
    )
    return has_content


def process_sentence(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    lemmas = [token.lemma_ for token in doc]
    return {
        "sentence": sentence,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas,
    }


def process_utterance(utterance):
    cleaned_text = clean_text(utterance["text"])
    sentences = sent_tokenize(cleaned_text)
    meaningful_sentences = [
        sent for sent in sentences if should_keep_sentence(sent)
    ]
    processed_sentences = [
        process_sentence(sent) for sent in meaningful_sentences
    ]
    return processed_sentences


class DataCleaningPipeline:
    def process(self, info: ProcessInformation):
        # 假设info.transcription是一个说话段落列表，每个元素有'text'字段
        if not hasattr(info, "transcription") or not info.transcription:
            return
        for utterance in info.transcription:
            utterance["processed"] = process_utterance(utterance)

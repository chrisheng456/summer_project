import json
import re
import spacy
from nltk.tokenize import sent_tokenize

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define filler words that have no actual meaning (to be removed)
PURE_FILLER_WORDS = {
    'uh', 'um', 'ah', 'er', 'hm', 'hmm', 'eh', 'mhm', 'uh-huh', 'uh-uh',
    'erm', 'ummm', 'uhm', 'mm', 'mmm', 'huh'
}

# Interjections with semantic meaning (to be kept)
MEANINGFUL_INTERJECTIONS = {
    'yes', 'no', 'yeah', 'yep', 'nope', 'okay', 'ok', 'right', 'oh', 'ah',
    'wow', 'great', 'good', 'fine', 'exactly', 'absolutely', 'certainly',
    'sure', 'agreed', 'understood', 'got it', 'please', 'thanks', 'thank you'
}

# Abbreviation expansion mapping
ABBREVIATIONS = {
    'btw': 'by the way', 'w/': 'with', 'w/o': 'without', 'e.g.': 'for example',
    'i.e.': 'that is', 'etc.': 'and so on', 'vs.': 'versus', 'approx.': 'approximately',
    '&': 'and', '+': 'plus', 're:': 'regarding', 'asap': 'as soon as possible',
    'afaik': 'as far as i know', 'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'fyi': 'for your information', 'tbh': 'to be honest', 'n/a': 'not applicable'
}

# Noise patterns to remove
NOISE_PATTERNS = [
    r'\([^)]*\)',  # remove parentheses
    r'\[[^\]]*\]',  # remove brackets
    r'\b(?:cough|laugh|laughter|applause|breath|sigh|noise|static)\b',  # remove noise tokens
    r'\b(?:\w*[\*#]\w*)\b'  # remove words with special characters
]


def clean_text(text):
    """
    Perform multi-step text cleaning:
    1. Expand abbreviations
    2. Remove meaningless filler words
    3. Remove noise tokens
    4. Normalize punctuation and whitespace
    """
    # Step 1: expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)

    # Step 2: remove meaningless filler words (keep meaningful interjections)
    filler_pattern = r'\b(?:' + '|'.join(map(re.escape, PURE_FILLER_WORDS)) + r')\b'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)

    # Step 3: remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Step 4: normalize punctuation and spacing
    text = re.sub(r'([,.?!;:])\1+', r'\1', text)  # collapse duplicate punctuation
    text = re.sub(r'\s+([,.?!;:])', r'\1', text)  # remove spaces before punctuation
    text = re.sub(r'([,.?!;:])(\w)', r'\1 \2', text)  # ensure space after punctuation
    text = re.sub(r'\s{2,}', ' ', text)  # collapse multiple spaces
    text = text.strip()

    # Ensure the text ends with proper punctuation
    if text and text[-1] not in {'.', '?', '!'}:
        text += '.'

    return text


def should_keep_sentence(sentence):
    """
    Determine whether a sentence should be kept (even if short).
    Criteria: must contain meaningful content words or interjections.
    """
    if not sentence.strip():
        return False

    doc = nlp(sentence)

    has_content = any(
        token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'} or
        token.text.lower() in MEANINGFUL_INTERJECTIONS
        for token in doc
    )
    return has_content


def process_sentence(sentence):
    """
    Process a single sentence with tokenization, POS tagging, and lemmatization.
    """
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    lemmas = [token.lemma_ for token in doc]

    return {
        "sentence": sentence,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas
    }


def process_utterance(utterance):
    """
    Process one utterance (speaker turn):
    1. Clean the text
    2. Segment into sentences
    3. Keep only sentences with meaningful content
    4. Run linguistic annotation (POS + lemma)
    """
    cleaned_text = clean_text(utterance["text"])

    # Sentence segmentation
    sentences = sent_tokenize(cleaned_text)

    # Keep meaningful sentences only
    meaningful_sentences = [sent for sent in sentences if should_keep_sentence(sent)]

    # Annotate each kept sentence
    processed_sentences = [process_sentence(sent) for sent in meaningful_sentences]

    return processed_sentences


def main(input_file, output_file):
    """
    Preprocess a diarized transcript JSON file:
    - Clean and filter each utterance
    - Annotate meaningful sentences
    - Save enriched JSON with 'processed' field added
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        diarized_data = json.load(f)

    processed_count = 0
    for utterance in diarized_data:
        processed = process_utterance(utterance)
        utterance["processed"] = processed
        processed_count += len(processed)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diarized_data, f, indent=2, ensure_ascii=False)

    print(f"Preprocessing complete! {len(diarized_data)} utterances processed.")
    print(f"Total {processed_count} meaningful sentences kept.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    input_json = "dataset/ICSI/diarized_json/Bdb001_diarized.json"
    output_json = "dataset/ICSI/processed_json/Bdb001_processed.json"
    main(input_json, output_json)

import json
import re
import spacy
from nltk.tokenize import sent_tokenize

# 加载spaCy英语模型
nlp = spacy.load("en_core_web_sm")

# 重新定义填充词列表 - 只包含无实际含义的填充词
PURE_FILLER_WORDS = {
    'uh', 'um', 'ah', 'er', 'hm', 'hmm', 'eh', 'mhm', 'uh-huh', 'uh-uh',
    'erm', 'ummm', 'uhm', 'mm', 'mmm', 'huh'
}

# 有实际含义的语气词 - 需要保留
MEANINGFUL_INTERJECTIONS = {
    'yes', 'no', 'yeah', 'yep', 'nope', 'okay', 'ok', 'right', 'oh', 'ah',
    'wow', 'great', 'good', 'fine', 'exactly', 'absolutely', 'certainly',
    'sure', 'agreed', 'understood', 'got it', 'please', 'thanks', 'thank you'
}

# 缩写替换表
ABBREVIATIONS = {
    'btw': 'by the way', 'w/': 'with', 'w/o': 'without', 'e.g.': 'for example',
    'i.e.': 'that is', 'etc.': 'and so on', 'vs.': 'versus', 'approx.': 'approximately',
    '&': 'and', '+': 'plus', 're:': 'regarding', 'asap': 'as soon as possible',
    'afaik': 'as far as i know', 'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'fyi': 'for your information', 'tbh': 'to be honest', 'n/a': 'not applicable'
}

# 噪音模式
NOISE_PATTERNS = [
    r'\([^)]*\)',  # 删除括号内容
    r'\[[^\]]*\]',  # 删除方括号内容
    r'\b(?:cough|laugh|laughter|applause|breath|sigh|noise|static)\b',  # 删除特定噪音词
    r'\b(?:\w*[\*#]\w*)\b'  # 删除含特殊符号的词
]


def clean_text(text):
    """
    执行多阶段文本清洗：
    1. 替换缩写
    2. 删除无意义填充词
    3. 删除噪音模式
    4. 规范化标点和空格
    """
    # 步骤1: 替换缩写
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)

    # 步骤2: 删除无意义填充词（保留有含义的语气词）
    filler_pattern = r'\b(?:' + '|'.join(map(re.escape, PURE_FILLER_WORDS)) + r')\b'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)

    # 步骤3: 删除噪音模式
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 步骤4: 规范化标点和空格
    text = re.sub(r'([,.?!;:])\1+', r'\1', text)  # 减少重复标点
    text = re.sub(r'\s+([,.?!;:])', r'\1', text)  # 移除标点前空格
    text = re.sub(r'([,.?!;:])(\w)', r'\1 \2', text)  # 添加标点后空格
    text = re.sub(r'\s{2,}', ' ', text)  # 减少连续空格
    text = text.strip()

    # 确保句末有标点
    if text and text[-1] not in {'.', '?', '!'}:
        text += '.'

    return text


def should_keep_sentence(sentence):
    """
    判断是否应保留句子（即使很短）
    """
    # 空句子直接跳过
    if not sentence.strip():
        return False

    # 处理后的句子
    doc = nlp(sentence)

    # 检查是否包含实际内容
    has_content = any(
        token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'} or
        token.text.lower() in MEANINGFUL_INTERJECTIONS
        for token in doc
    )

    return has_content


def process_sentence(sentence):
    """
    处理单个句子：分词、词性标注、词形还原
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
    处理单个说话段落：
    1. 文本清洗
    2. 语义分段
    3. 保留有实际含义的短句
    4. 语言学标注
    """
    # 文本清洗
    cleaned_text = clean_text(utterance["text"])

    # 语义分段
    sentences = sent_tokenize(cleaned_text)

    # 保留有实际含义的句子
    meaningful_sentences = [sent for sent in sentences if should_keep_sentence(sent)]

    # 处理每个有意义的句子
    processed_sentences = [process_sentence(sent) for sent in meaningful_sentences]

    return processed_sentences


def main(input_file, output_file):
    # 读取说话人分离后的JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        diarized_data = json.load(f)

    # 处理每个说话段落
    processed_count = 0
    for utterance in diarized_data:
        processed = process_utterance(utterance)
        utterance["processed"] = processed

        # 统计实际处理的句子数
        processed_count += len(processed)

    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diarized_data, f, indent=2, ensure_ascii=False)

    print(f"预处理完成！处理了 {len(diarized_data)} 个说话段落")
    print(f"共保留 {processed_count} 个有意义的句子")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    input_json = "dataset/ICSI/diarized_json/Bdb001_diarized.json"
    output_json = "dataset/ICSI/processed_json/Bdb001_processed.json"
    main(input_json, output_json)
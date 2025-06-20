# typesense_utils.py
# 纯 Python 关键词过滤，无需依赖外部全文检索服务

import json
from pathlib import Path


def search_keyword_ts(query: str, top_k: int = 50):
    """
    在 post_process_local.py 生成的 *_local.json 中的 lines 列表里，
    用简单的子字符串匹配找到包含关键词的条目，
    返回最多 top_k 项 (id, text)

    :param query: 搜索关键词
    :param top_k: 最大返回候选数
    :return: List of (id, text)
    """
    # 找到最新生成的 *_local.json 文件
    files = sorted(Path('.').glob('meeting_minutes_*_local.json'))
    if not files:
        return []
    fp = files[-1]

    # 读取 JSON
    data = json.loads(fp.read_text(encoding='utf-8'))
    hits = []
    # 对每一行做子字符串匹配
    for idx, line in enumerate(data.get('lines', [])):
        text = line.get('text', '').strip()
        if not text:
            continue
        if query.lower() in text.lower():
            # 构造唯一 ID
            doc_id = f"{fp.stem}_{idx}"
            hits.append((doc_id, text))
            if len(hits) >= top_k:
                break
    return hits

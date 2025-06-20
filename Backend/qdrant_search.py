# qdrant_search.py
# 功能：先关键词过滤，再向量重排，实现混合检索

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typesense_utils import search_keyword_ts

# 配置
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 向量编码模型
COLLECTION_NAME = "meeting_minutes"
TS_TOPK = 50    # 全文过滤候选数
FINAL_K = 5      # 最终返回数

class QdrantSearch:
    def __init__(self,
                 model_name: str = MODEL_NAME,
                 ts_topk: int = TS_TOPK,
                 final_k: int = FINAL_K,
                 col: str = COLLECTION_NAME):
        # 初始化 Typesense 全文检索和 Qdrant 向量检索
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient()
        self.col    = col
        self.ts_topk = ts_topk
        self.final_k = final_k

    def search(self, query: str):
        # 第一步：全文关键词过滤，得到候选 (id, text)
        candidates = search_keyword_ts(query, top_k=self.ts_topk)
        if not candidates:
            return []
        ids, texts = zip(*candidates)

        # 第二步：向量编码 & 相似度计算
        q_vec    = self.model.encode(query)
        doc_vecs = self.model.encode(list(texts))
        # 计算余弦相似度
        q_norm   = np.linalg.norm(q_vec)
        d_norms  = np.linalg.norm(doc_vecs, axis=1)
        sims = (doc_vecs @ q_vec) / (d_norms * q_norm + 1e-10)

        # 第三步：排序并取 top final_k
        idxs = np.argsort(-sims)[:self.final_k]
        results = []
        for i in idxs:
            results.append({
                'id': ids[i],
                'text': texts[i],
                'score': float(sims[i])
            })
        return results

if __name__ == '__main__':
    qs = QdrantSearch()
    kw = input("请输入关键词：").strip()
    hits = qs.search(kw)
    if not hits:
        print("未找到匹配结果")
    else:
        print(f"共匹配到 {len(hits)} 条结果：")
        for hit in hits:
            print(f"[{hit['score']:.4f}] {hit['id']} -> {hit['text']}")

# qdrant_search.py
# 混合检索：关键词命中排前，其它按向量相似度排序

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typesense_utils import search_keyword_ts

# 配置
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
COLLECTION = "meeting_minutes"
SUBSET_K   = 50   # 向量检索候选数
FINAL_K    = 5    # 最终返回数

class HybridSearcher:
    def __init__(self):
        # 初始化模型和 Qdrant 客户端
        self.model  = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient()

    def search(self, query: str) -> list:
        # 1) 关键词过滤，拿到所有命中的 (doc_id, text)
        kw_hits = search_keyword_ts(query, top_k=SUBSET_K)
        # 提取出整数 ID
        kw_ids = {
            int(doc_id.rsplit('_', 1)[-1])
            for doc_id, _ in kw_hits
            if '_' in doc_id and doc_id.rsplit('_', 1)[-1].isdigit()
        }

        # 2) 向量检索，获取前 SUBSET_K 个命中及其分数
        q_vec = self.model.encode(query)
        resp = self.client.search(
            collection_name=COLLECTION,
            query_vector=q_vec.tolist(),
            limit=SUBSET_K
        )
        vector_hits = [
            {"id": r.id, "text": r.payload.get("text", ""), "score": r.score}
            for r in resp
        ]

        # 3) 排序：先放关键词命中，不考虑向量分数；其他按 score 降序
        # 1. 用 kw_hits 的次序决定 prioritized
        prioritized = []
        for doc_id, text in kw_hits:
            idx = int(doc_id.rsplit('_', 1)[-1])
            # 在 vector_hits 里找这个 idx 对应的文本
            match = next((h for h in vector_hits if h["id"] == idx), None)
            if match:
                prioritized.append(match)

        # 2. 其它按分数排列
        others = [h for h in vector_hits if h["id"] not in {h["id"] for h in prioritized}]
        others.sort(key=lambda x: x["score"], reverse=True)

        ordered = prioritized + others

        # 4) 合并并截取前 FINAL_K 条，同时去掉 score
        ordered = prioritized + others
        results = [{"id": h["id"], "text": h["text"]} for h in ordered[:FINAL_K]]
        return results

if __name__ == '__main__':
    hs = HybridSearcher()
    query = input("🔍 请输入搜索关键词：").strip()
    results = hs.search(query)
    if not results:
        print("❌ 未找到匹配结果。")
    else:
        print(f"✅ 找到 {len(results)} 条结果：\n")
        for r in results:
            print(f"{r['text']}")



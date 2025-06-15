# evaluate_retrieval.py

import json
from typing import List, Dict

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ─── 评测指标 ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / k

def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / len(relevant) if relevant else 0.0

def average_precision(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    AP@k = 平均的 P@i，其中 i 是每个相关文档在 top-k 中出现的位置
    """
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    DCG = sum( rel_i / log2(i+1) ), IDCG 为最理想情况下的 DCG
    相关性 rel_i 在这里二值化：相关=1，否则=0
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / np.log2(i + 1)
    # 理想 DCG：对 relevant 文档按最优顺序打分
    ideal_rels = [1]*min(len(relevant), k)
    idcg = sum(rel / np.log2(i+1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0

def mean_reciprocal_rank(retrieved: List[int], relevant: List[int]) -> float:
    """
    MRR = 平均( 1 / 排名位置 )，只计算第一个命中的位置
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

# ─── 检索函数 ──────────────────────────────────────────────────────────

class Retriever:
    def __init__(self,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "meeting_minutes"):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.model  = SentenceTransformer(model_name)
        self.col    = collection_name

    def retrieve(self, query: str, top_k: int = 10) -> List[int]:
        q_vec = self.model.encode(query)
        resp  = self.client.search(
            collection_name=self.col,
            query_vector=q_vec.tolist(),
            limit=top_k
        )
        # 返回命中文档的 ID 列表
        return [hit.id for hit in resp]

# ─── 主流程 ──────────────────────────────────────────────────────────

def evaluate(test_file: str,
             retriever: Retriever,
             ks: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    对测试集中的每个 query，统计 Precision@k / Recall@k / AP@k / nDCG@k / MRR
    最后汇总为一个 DataFrame，方便比较不同模型或参数
    """
    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)

    records: List[Dict] = []
    for item in tests:
        q         = item["query"]
        relevant  = item["relevant_ids"]
        retrieved = retriever.retrieve(q, top_k=max(ks))

        for k in ks:
            records.append({
                "query": q,
                "k": k,
                "Precision": precision_at_k(retrieved, relevant, k),
                "Recall": recall_at_k(retrieved, relevant, k),
                "AP": average_precision(retrieved, relevant, k),
                "nDCG": ndcg_at_k(retrieved, relevant, k),
                "MRR": mean_reciprocal_rank(retrieved, relevant),
            })

    df = pd.DataFrame(records)
    # 按模型聚合平均值
    summary = df.groupby("k").mean()[["Precision", "Recall", "AP", "nDCG", "MRR"]]
    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True,
                        help="ground_truth.json 的路径")
    parser.add_argument("--model", type=str, default="all-MiniLM-L12-v2",
                        help="sentence-transformers 模型名称")
    parser.add_argument("--col",   type=str, default="meeting_minutes",
                        help="Qdrant collection 名称")
    args = parser.parse_args()

    retriever = Retriever(
        model_name=args.model,
        collection_name=args.col
    )

    print(f"\n=== Evaluating model = {args.model} ===")
    summary = evaluate(args.test, retriever, ks=[1,3,5,10])
    print(summary.to_markdown(tablefmt="github"))
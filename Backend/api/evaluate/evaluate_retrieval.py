import json
from typing import List, Dict

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ─── Evaluation Metrics ────────────────────────────────────────────────────

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / k


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / len(relevant) if relevant else 0.0


def average_precision(retrieved: List[int], relevant: List[int], k: int) -> float:
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / np.log2(i + 1)
    ideal_rels = [1] * min(len(relevant), k)
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved: List[int], relevant: List[int]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


# ─── Retriever Definition ─────────────────────────────────────────────────

class Retriever:
    def __init__(self,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 #model_name: str = "all-MiniLM-L6-v2",
                 model_name: str = "all-mpnet-base-v2",
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
        return [hit.id for hit in resp]


# ─── Evaluation Workflow ──────────────────────────────────────────────────

def evaluate(test_file: str,
             retriever: Retriever,
             ks: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    with open(test_file, 'r', encoding='utf-8') as f:
        tests = json.load(f)

    records: List[Dict] = []
    for item in tests:
        q        = item['query']
        relevant = item['relevant_ids']
        retrieved = retriever.retrieve(q, top_k=max(ks))

        for k in ks:
            records.append({
                'query': q,
                'k': k,
                'Precision': precision_at_k(retrieved, relevant, k),
                'Recall': recall_at_k(retrieved, relevant, k),
                'AP': average_precision(retrieved, relevant, k),
                'nDCG': ndcg_at_k(retrieved, relevant, k),
                'MRR': mean_reciprocal_rank(retrieved, relevant),
            })

    df = pd.DataFrame(records)
    # Use numeric_only=True to avoid type conversion issues
    summary = df.groupby('k').mean(numeric_only=True)[['Precision', 'Recall', 'AP', 'nDCG', 'MRR']]
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, required=True,
                        help='Path to ground_truth.json')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--col', type=str, default='meeting_minutes',
                        help='Qdrant collection name')
    args = parser.parse_args()

    retriever = Retriever(
        model_name=args.model,
        collection_name=args.col
    )

    print(f"\n=== Evaluating model = {args.model} ===")
    summary = evaluate(args.test, retriever, ks=[1,3,5,10])
    print(summary.to_markdown(tablefmt='github'))

# index_msmarco.py  ✨可配置版
import argparse, pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# ─── CLI 参数 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--col',   default='msmarco_passages',
                    help='Qdrant collection name')
parser.add_argument('--model', default='all-MiniLM-L6-v2',
                    help='Sentence-Transformers model id')
parser.add_argument('--file',  default='collection.tsv',
                    help='TSV with columns: pid \\t passage')
parser.add_argument('--batch', type=int, default=512,
                    help='Batch size for upsert')
args = parser.parse_args()

# ─── 1. 读语料 ─────────────────────────────────────────────────────────────
df = pd.read_csv(args.file, sep='\t', names=['pid', 'text'],
                 dtype={'pid': int, 'text': str}, quoting=3)
print(f'Loaded {len(df):,} passages from {args.file}')

# ─── 2. 加载模型 ───────────────────────────────────────────────────────────
model = SentenceTransformer(args.model)
dim   = model.get_sentence_embedding_dimension()
print(f'Using model {args.model}  (dim = {dim})')

# ─── 3. 准备 Qdrant ────────────────────────────────────────────────────────
client = QdrantClient(host='localhost', port=6333)

# 如已存在同名 collection，可以带 --force 重建，或自己改逻辑
client.recreate_collection(
    collection_name=args.col,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
)

# ─── 4. 批量写入 ──────────────────────────────────────────────────────────
for i in range(0, len(df), args.batch):
    chunk = df.iloc[i:i+args.batch]
    vecs  = model.encode(chunk.text.tolist(), show_progress_bar=False)
    pts   = [PointStruct(id=int(pid), vector=v.tolist(), payload={'text': t})
             for pid, v, t in zip(chunk.pid, vecs, chunk.text)]
    client.upsert(collection_name=args.col, points=pts)

print(f'✅ Indexed {len(df):,} passages into "{args.col}"')

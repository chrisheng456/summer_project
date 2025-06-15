# index_msmarco.py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. 读 collection.tsv
df = pd.read_csv("collection.tsv", sep="\t", names=["pid","text"],
                 dtype={"pid":int,"text":str}, quoting=3)

# 2. 建 collection（384 维，COSINE）
client = QdrantClient()
client.recreate_collection(
  name="msmarco_passages",
  vector_params=VectorParams(size=384, distance=Distance.COSINE)
)

# 3. 选模型并 encode
model = SentenceTransformer("all-MiniLM-L6-v2")  # 或其他模型
batch_size = 512

# 4. 分批 upsert
for start in range(0, len(df), batch_size):
  batch = df.iloc[start:start+batch_size]
  texts = batch.text.tolist()
  pids  = batch.pid.tolist()
  vecs  = model.encode(texts, show_progress_bar=False)
  points = [
    PointStruct(id=pid, vector=vec.tolist(), payload={"text":txt})
    for pid, vec, txt in zip(pids, vecs, texts)
  ]
  client.upsert(collection_name="msmarco_passages", points=points)

print("✅ MS MARCO passages indexed:", len(df))
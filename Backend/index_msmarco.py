# index_msmarco.py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. 读 MS MARCO passages 文件
#    假设已经导出 dev-small 子集至 collection.tsv
#    格式：pid \t passage_text

df = pd.read_csv(
    "collection.tsv",
    sep="\t",
    names=["pid", "text"],
    dtype={"pid": int, "text": str},
    quoting=3
)
print(f"Loaded {len(df)} passages for indexing")

# 2. 载入模型并获取向量维度
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_id)
dim = model.get_sentence_embedding_dimension()
print(f"Using model {model_id} with embedding dimension = {dim}")

# 3. 初始化 Qdrant 客户端
client = QdrantClient(host="localhost", port=6333)

# 4. (重)建 collection，使用动态维度
client.recreate_collection(
    collection_name="msmarco_passages",
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
)

# 5. 分批 upsert 到 Qdrant
batch_size = 512
for start in range(0, len(df), batch_size):
    batch = df.iloc[start : start + batch_size]
    texts = batch.text.tolist()
    pids = batch.pid.tolist()
    vectors = model.encode(texts, show_progress_bar=False)
    points = [
        PointStruct(id=pid, vector=vec.tolist(), payload={"text": txt})
        for pid, vec, txt in zip(pids, vectors, texts)
    ]
    client.upsert(collection_name="msmarco_passages", points=points)

print(f"✅ Indexed {len(df)} MS MARCO passages into 'msmarco_passages'.")

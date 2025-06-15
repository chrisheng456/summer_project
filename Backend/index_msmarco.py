# index_msmarco.py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. 读 MS MARCO passages 文件
#    假设你已用 ir_datasets 导出： collection.tsv
#    内容格式：pid \t passage_text

df = pd.read_csv(
    "collection.tsv",
    sep="\t",
    names=["pid", "text"],
    dtype={"pid": int, "text": str},
    quoting=3  # 不去除引号
)
print(f"Loaded {len(df)} passages for indexing")

# 2. 初始化 Qdrant 客户端
client = QdrantClient(host="localhost", port=6333)

# 3. (重)建 collection，注意参数名称 vectors_config
client.recreate_collection(
    collection_name="msmarco_passages",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 4. 载入模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. 分批 upsert 到 Qdrant
batch_size = 512
for start in range(0, len(df), batch_size):
    batch = df.iloc[start : start + batch_size]
    texts = batch.text.tolist()
    pids = batch.pid.tolist()
    # 编码
    vectors = model.encode(texts, show_progress_bar=False)
    points = [
        PointStruct(id=pid, vector=vec.tolist(), payload={"text": txt})
        for pid, vec, txt in zip(pids, vectors, texts)
    ]
    client.upsert(collection_name="msmarco_passages", points=points)

print(f"✅ Indexed {len(df)} MS MARCO passages into 'msmarco_passages'.")

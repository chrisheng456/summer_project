# qdrant_utils.py
# 功能：将长转录文本拆分成短段，并索引到 Qdrant，使用方案 B：指定 HNSW 参数（HnswConfigDiff）

import json
import re
import textwrap
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import HnswConfigDiff
from sentence_transformers import SentenceTransformer

# 配置参数
MODEL_ID        = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "meeting_minutes"
BATCH_SIZE      = 512
CHUNK_SIZE      = 300  # 超长段再拆分的最大长度
HNSW_EF         = 200
HNSW_M          = 48
FULL_SCAN_THRESH = 10000  # HNSW 全扫描阈值


def upload_meeting_minutes_vectors():
    # 1. 找到最新的 本地转录 JSON 文件
    fp = sorted(Path('.').glob('meeting_minutes_*_local.json'))[-1]
    data = json.loads(fp.read_text(encoding='utf-8'))
    meeting_id = fp.stem
    raw_text = data.get('transcription', '')

    # 2. 文本拆分：先按中英文句号、感叹问号等切句，再对过长句子按固定长度拆分
    segments = []
    for sent in re.split(r'(?<=[。！？\.!?])\s*', raw_text):
        s = sent.strip()
        if not s:
            continue
        if len(s) <= CHUNK_SIZE:
            segments.append(s)
        else:
            segments.extend(textwrap.wrap(s, CHUNK_SIZE))
    print(f"→ Splitted into {len(segments)} segments for indexing.")

    # 3. 初始化模型和 Qdrant 客户端
    model = SentenceTransformer(MODEL_ID)
    dim   = model.get_sentence_embedding_dimension()
    client = QdrantClient()

    # 4. (重)建 collection，指定 HNSWConfigDiff 参数
    vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
    hnsw_config    = HnswConfigDiff(
        m=HNSW_M,
        ef_construct=HNSW_EF,
        full_scan_threshold=FULL_SCAN_THRESH
    )
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
        hnsw_config=hnsw_config
    )

    # 5. 批量编码并上传
    points = []
    for idx, text in enumerate(segments):
        vec = model.encode(text)
        uid = idx
        points.append(PointStruct(id=uid, vector=vec.tolist(), payload={'text': text}))
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Indexed {len(segments)} segments into '{COLLECTION_NAME}'.")


if __name__ == '__main__':
    upload_meeting_minutes_vectors()

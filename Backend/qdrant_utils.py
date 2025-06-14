from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import uuid
from pathlib import Path

def upload_meeting_minutes_vectors():
    # 连接本地 Qdrant
    client = QdrantClient("localhost", port=6333)

    # 创建或重建集合
    client.recreate_collection(
        collection_name="meeting_minutes",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ 已创建或重建集合 meeting_minutes")

    # 加载 SentenceTransformer 模型
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # 找到最新的会议纪要 JSON 文件
    files = sorted(Path(".").glob("meeting_minutes_*_local.json"))
    if not files:
        print("❌ 未找到会议纪要 JSON 文件！")
        return

    file = files[-1]
    data = json.loads(file.read_text(encoding="utf-8"))

    # 准备要编码的文本（摘要 + 每个行动项的 task）
    texts = [data.get("abstract_summary", "")] + [item.get("task", "") for item in data.get("action_items", [])]
    print(f"🔄 编码文本数量: {len(texts)}")

    # 生成向量
    vectors = encoder.encode(texts)

    # 构建上传点列表
    points = []
    for vec, text in zip(vectors, texts):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": text}
        )
        points.append(point)

    # 上传向量
    client.upsert(collection_name="meeting_minutes", points=points)
    print(f"✅ 上传完成，文件: {file.name}")

if __name__ == "__main__":
    upload_meeting_minutes_vectors()

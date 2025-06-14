from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def vector_search():
    # 连接本地 Qdrant 服务
    client = QdrantClient(host="localhost", port=6333)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    query = input("🔍 请输入搜索内容：").strip()
    if not query:
        print("❌ 搜索内容不能为空！")
        return

    # 查询向量化
    query_vector = encoder.encode(query).tolist()

    # 语义搜索，返回 top 5 相关内容
    results = client.search(
        collection_name="meeting_minutes",
        query_vector=query_vector,
        limit=5
    )

    if not results:
        print("❌ 未找到相关内容")
        return

    print("\n🔎 搜索结果：\n")
    for hit in results:
        # 假设向量上传时，payload 中存了 text 字段，你可以根据你实际保存的字段名调整这里
        text = hit.payload.get('text', '[无文本内容]')
        print(f"📌 相关文本: {text}")
        print(f"   相似度分数: {hit.score:.4f}\n")

if __name__ == "__main__":
    vector_search()

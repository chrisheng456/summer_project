from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 初始化 Qdrant 客户端
client = QdrantClient("localhost", port=6333)

# 加载嵌入模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 定义集合名
collection_name = "meeting_minutes"

# 封装搜索方法（兼容新版和旧版）
def smart_search(client, collection_name, query_vector, top_k=5):
    try:
        # 新版本接口
        return client.search_points(
            collection_name=collection_name,
            query_vector=query_vector,
            top=top_k,
            with_payload=True
        )
    except AttributeError:
        # 老版本接口
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

# 主函数
def vector_search():
    query = input("🔍 请输入搜索内容：")
    query_vector = model.encode(query).tolist()

    try:
        results = smart_search(client, collection_name, query_vector)
    except Exception as e:
        print("❌ 搜索失败:", str(e))
        return

    print("\n🔎 搜索结果：\n")
    for point in results:
        payload = point.payload
        score = point.score
        if payload and "text" in payload:
            print(f"📌 相关文本: {payload['text']}")
        else:
            print("📌 相关文本: [无文本数据]")
        print(f"   相似度分数: {score:.4f}\n")

# 执行
if __name__ == "__main__":
    vector_search()

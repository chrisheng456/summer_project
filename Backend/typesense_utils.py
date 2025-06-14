import typesense
import json
import argparse
from pathlib import Path
import sys
import re

# 初始化 Typesense 客户端
client = typesense.Client({
    'nodes': [{
        'host': 'localhost',
        'port': 8108,
        'protocol': 'http'
    }],
    'api_key': 'xyz',  # ← 请替换为你实际设置的 API 密钥
    'connection_timeout_seconds': 2
})

# 创建集合（只需一次）
def create_collection():
    schema = {
        "name": "meeting_minutes",
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "title", "type": "string"},
            {"name": "transcription", "type": "string"},
            {"name": "abstract_summary", "type": "string"},
            {"name": "key_points", "type": "string[]"},
            {"name": "action_items", "type": "string[]"},
            {"name": "sentiment", "type": "string"},
            {"name": "participants", "type": "string[]", "facet": True},  # 新增
            {"name": "meeting_tags", "type": "string[]", "facet": True},  # 新增
            {"name": "date", "type": "string", "facet": True}
        ]
    }
    try:
        client.collections.create(schema)
        print("✅ 集合已成功创建")
    except Exception as e:
        print("⚠️ 集合可能已存在:", e)

# 上传最新的纪要
def upload_latest_minute():
    # 查找最新的会议纪要 JSON 文件
    files = sorted(Path(".").glob("meeting_minutes_*_local.json"))
    if not files:
        print("❌ 没有找到会议纪要 JSON 文件")
        return

    file = files[-1]
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = {
        "id": file.stem,
        "title": "会议纪要 - " + file.stem,
        "transcription": data.get("transcription", ""),
        "abstract_summary": data.get("abstract_summary", ""),
        "key_points": data.get("key_points", []),
        "action_items": [item["task"] for item in data.get("action_items", [])],
        "sentiment": data.get("sentiment", ""),
        "participants": data.get("participants", []),
        "meeting_tags": data.get("tags", []),
        "date": file.stem.split("_")[2]
    }

    try:
        client.collections["meeting_minutes"].documents.create(doc)
        print(f"✅ 上传成功: {file.name}")
    except Exception as e:
        print("❌ 上传失败:", e)

# 搜索纪要
def search(query):
    results = client.collections["meeting_minutes"].documents.search({
        "q": query,
        "query_by": "abstract_summary,action_items,key_points,transcription",
        'num_typos': 0
    })
    print(f"🔍 共找到 {len(results['hits'])} 条结果：\n")
    for hit in results["hits"]:
        doc = hit["document"]
        print(f"📌 {doc['title']}")
        print(f"📝 摘要: {doc['abstract_summary']}\n")

def highlight(text, keyword):
    # 高亮关键词
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(f"\033[1;31m{keyword}\033[0m", text)  # 红色高亮，适用于终端

def search(query, exact=False):
    search_params = {
        'q': query,
        'query_by': "abstract_summary,action_items,key_points,transcription",
        'per_page': 5,
    }

    if exact:
        search_params['num_typos'] = 0  # 精确匹配，不允许模糊
    else:
        search_params['num_typos'] = 2  # 允许最多2个字符错位（默认）

    try:
        results = client.collections['meeting_minutes'].documents.search(search_params)

        if results['found'] == 0:
            print(f"\n🔍 没有找到匹配结果（{'精确' if exact else '模糊'}搜索）")
            return

        print(f"\n🔍 共找到 {results['found']} 条结果：\n")

        for hit in results["hits"]:
            doc = hit["document"]
            score = hit["text_match"]
            print(f"📌 {doc['title']}  (score: {score})")
            print("📝 摘要:", highlight(doc["abstract_summary"], query))
            if doc.get("key_points"):
                print("📌 关键点:", [highlight(kp, query) for kp in doc["key_points"]])
            if doc.get("action_items"):
                print("✅ 行动项:", [highlight(ai, query) for ai in doc["action_items"]])
            print("🧾 转录片段:", highlight(doc["transcription"][:300], query), "...")
            print("-" * 60)

    except Exception as e:
        print("❌ 搜索失败:", e)

# 命令行入口
if __name__ == "__main__":
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "create":
        create_collection()
    elif command == "upload":
        upload_latest_minute()
    elif command == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        exact_mode = "--exact" in sys.argv
        search(query, exact=exact_mode)
    else:
        print("用法：python typesense_utils.py [create|upload|search <query> [--exact]]")


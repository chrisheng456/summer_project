# make_collection_dev.py

import ir_datasets

# 1. 加载 dev-small 的 qrels，收集所有正例 pid
dev = ir_datasets.load("msmarco-passage/dev/small")
relevant_pids = sorted({int(q.passage_id) for q in dev.qrels_iter() if q.relevance > 0})
print(f"→ Found {len(relevant_pids)} PIDs in dev/small.")

# 2. 加载完整的文档存储（ir_datasets 会自动用本地缓存或在线拉取）
dataset = ir_datasets.load("msmarco-passage")

# 3. 写入 collection_dev.tsv
with open("collection_dev.tsv", "w", encoding="utf-8") as fout:
    for pid in relevant_pids:
        doc = dataset.docs_store.get(str(pid))
        fout.write(f"{pid}\t{doc.text.replace(chr(10), ' ')}\n")

print("✅ Wrote collection_dev.tsv with", len(relevant_pids), "lines.")
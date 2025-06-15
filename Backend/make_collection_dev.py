# make_collection_dev.py

import ir_datasets

# 1. 加载 dev-small qrels，收集所有正例 pid
print("Loading dev-small qrels...")
dev = ir_datasets.load("msmarco-passage/dev/small")
relevant_pids = sorted({int(q.doc_id) for q in dev.qrels_iter() if q.relevance > 0})
print(f"→ Got {len(relevant_pids)} PIDs from dev-small.")

# 2. 遍历 dev-small docs_iter，只写相关的
with open("collection_dev.tsv", "w", encoding="utf-8") as fout:
    for doc in dev.docs_iter():
        pid = int(doc.doc_id)
        if pid in relevant_pids:
            text = doc.text.replace("\n", " ")
            fout.write(f"{pid}\t{text}\n")

print(f"✅ Wrote collection_dev.tsv with {len(relevant_pids)} lines.")
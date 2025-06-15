# make_ground_truth.py
import pandas as pd
import json

# 1. 读 queries.tsv （仍然是制表符分隔）
df_q = pd.read_csv("queries.tsv", sep="\t", names=["qid", "query"])

# 2. 读 qrels.tsv —— 用正则空白分隔四列：qid, dummy, pid, rel
df_r = pd.read_csv(
    "qrels.tsv",
    sep=r"\s+",
    names=["qid", "_dummy", "pid", "rel"],
    engine="python"
)

# 3. 转成整型，过滤出 rel>0
df_r["rel"] = df_r["rel"].astype(int)
df_pos = df_r[df_r["rel"] > 0]

# 4. 构建 ground_truth 列表
gt = []
for qid, grp in df_pos.groupby("qid"):
    # 查回 query 文本
    query_text = df_q.loc[df_q.qid == qid, "query"].iloc[0]
    # 所有相关的 doc id
    pids = grp["pid"].astype(int).tolist()
    gt.append({
        "query": query_text,
        "relevant_ids": pids
    })

# 5. 写出 JSON
with open("ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

print(f"✅ Generated ground_truth.json with {len(gt)} entries.")
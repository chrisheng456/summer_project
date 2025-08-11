# batch_benchmark.py  ← 覆盖原文件
import subprocess, shlex, sys, io, re, pandas as pd, time

MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]

COL_TPL = "msmarco_{name}"
PY = shlex.quote(sys.executable)          # 当前虚拟环境 python

def run(cmd: str):
    print(f"\n$ {cmd}")
    t0 = time.time()
    out = subprocess.check_output(shlex.split(cmd), text=True)
    print(f"  ✅  {time.time()-t0:6.1f}s")
    return out

# ---------- 关键修正：更健壮地解析 ASCII 表 ----------
def parse_table(text: str, model: str) -> pd.DataFrame:
    rows = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("|")]
    if len(rows) < 3:
        raise ValueError("表格行太少，无法解析")

    header = [c.strip() for c in rows[0].split("|")[1:-1]]
    data   = [r.split("|")[1:-1] for r in rows[2:]]  # 跳过分隔行 rows[1]

    df = pd.DataFrame(data, columns=header)
    df = df.replace(r"\s+", "", regex=True)          # 去掉空格
    df = df.apply(pd.to_numeric, errors="ignore")    # 数值列转成数值
    df.insert(0, "model", model.split("/")[-1])
    return df
# -----------------------------------------------------

all_rows = []
for m in MODELS:
    col = COL_TPL.format(name=m.split("/")[-1])

    run(f"{PY} index_msmarco.py --model {m} --col {col}")
    out = run(f"{PY} evaluate_retrieval.py --test ground_truth.json --model {m} --col {col}")

    all_rows.append(parse_table(out, m))

benchmark = pd.concat(all_rows, ignore_index=True)
benchmark.to_csv("benchmark.csv", index=False)

print("\n🎉  All done. 结果写入 benchmark.csv\n")
print(benchmark.to_markdown(index=False))

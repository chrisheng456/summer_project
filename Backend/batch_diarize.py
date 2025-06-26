# batch_diarize.py

import os
import subprocess
import sys         # <-- 新增

# wav 存放的根目录
WAV_ROOT   = "dataset/ICSI/Signals"
# 原始 transcript JSON
TRANS_ROOT = "dataset/ICSI/icsi_json"
# 输出 diary JSON
OUT_ROOT   = "dataset/ICSI/diarized_json"
os.makedirs(OUT_ROOT, exist_ok=True)

# 用当前运行的解释器
PYTHON = sys.executable    # <— 这样一定是 .venv 里的 python
TEST_ONLY = ["Bdb001"]

for root, _, files in os.walk(WAV_ROOT):
    for fname in files:
        if not fname.endswith(".wav"):
            continue
        wav_path = os.path.join(root, fname)
        base = fname.replace(".interaction.wav","").replace(".wav","")
        in_json  = os.path.join(TRANS_ROOT, f"{base}.json")
        out_json = os.path.join(OUT_ROOT,  f"{base}_diarized.json")

        if not os.path.exists(in_json):
            print(f"⚠️ 跳过，没有找到 transcript: {in_json}")
            continue

        cmd = [
            PYTHON,                       # 用虚拟环境里的 Python
            "diarize_and_align.py",      # 因为你现在在 Backend 目录下
            wav_path, in_json, out_json
        ]
        print("🛠️ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"✅ Done {base}")

print("🎉 全部对齐完成，结果在", OUT_ROOT)

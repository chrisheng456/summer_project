#!/usr/bin/env python3
# diarize_and_align.py

import os
# 禁用 symlink，以避免 Windows 下权限问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import json
import time
import torch
from pyannote.audio import Pipeline

def load_transcript(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_transcript(json_path, transcript):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

def align_speakers(wav_path, transcript_json_in, transcript_json_out):
    # 1) 选设备：优先 GPU，否则 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"→ Using device: {device}")

    # 2) 从环境变量读取 HF 访问令牌
    token = (
        os.environ.get("HF_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "请先在环境变量中设置 HF_HUB_TOKEN（或 HUGGINGFACE_TOKEN / HUGGINGFACE_HUB_TOKEN）"
        )

    # 3) 加载说话人分离 pipeline，并移动到 device
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token
    )
    pipeline.to(device)

    # 4) 运行 diarization
    start = time.perf_counter()
    diarization = pipeline(wav_path)
    elapsed = time.perf_counter() - start
    print(f"→ Diarization done in {elapsed/60:.2f} minutes")

    # 5) 提取 (start, end, label) 列表
    diar_segments = [
        (segment.start, segment.end, label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]

    # 6) 读取原始转写 JSON
    transcript = load_transcript(transcript_json_in)

    # 7) 对齐每条 utterance 到最佳 speaker label
    for utt in transcript:
        if "offset" in utt and "duration" in utt:
            start_sec = utt["offset"] / 1e7
            end_sec   = start_sec + utt["duration"] / 1e7
        elif "start" in utt and "end" in utt:
            start_sec = utt["start"]
            end_sec   = utt["end"]
        else:
            continue

        best_label  = "Unknown"
        max_overlap = 0.0
        for seg_start, seg_end, label in diar_segments:
            overlap = max(0.0, min(end_sec, seg_end) - max(start_sec, seg_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_label  = label

        utt["speaker"] = best_label

    # 8) 保存对齐后结果
    save_transcript(transcript_json_out, transcript)
    print(f"→ Saved diarized transcript to {transcript_json_out}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diarize_and_align.py WAV_PATH IN_JSON OUT_JSON")
        sys.exit(1)

    wav_path, transcript_in, transcript_out = sys.argv[1:]
    align_speakers(wav_path, transcript_in, transcript_out)

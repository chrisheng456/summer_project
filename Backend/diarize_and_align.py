# diarize_and_align.py

import os
# 禁用 symlink，以免 Windows 权限不足报错
os.environ["HF_HUB_DISABLE_SYMLINKS"]        = "1"
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

    # 2) 加载 pipeline（不含 device），再 to(device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=True
    )
    pipeline.to(device)

    # 3) 运行说话人分离
    start = time.perf_counter()
    diarization = pipeline(wav_path)
    elapsed = time.perf_counter() - start
    print(f"→ Diarization done in {elapsed/60:.2f} minutes")

    # 4) 转成列表 (start_sec, end_sec, speaker_label)
    diar_segments = [
        (segment.start, segment.end, label)
        for segment, _, label in diarization.itertracks(yield_label=True)
    ]

    # 5) 读取原始转写
    transcript = load_transcript(transcript_json_in)

    # 6) 对齐每条 utterance 到 best_label
    for utt in transcript:
        # 支持两种格式：Azure 的 offset/duration（100ns ticks），
        # 也支持 ICSI JSON 的 start/end（秒）
        if "offset" in utt and "duration" in utt:
            start_sec = utt["offset"]  / 1e7
            end_sec   = start_sec + utt["duration"] / 1e7
        elif "start" in utt and "end" in utt:
            start_sec = utt["start"]
            end_sec   = utt["end"]
        else:
            # 无法识别的格式
            continue

        best_label  = "Unknown"
        max_overlap = 0.0
        for seg_start, seg_end, label in diar_segments:
            overlap = max(0.0, min(end_sec, seg_end) - max(start_sec, seg_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_label  = label

        utt["speaker"] = best_label

    # 7) 保存带 speaker 的 JSON
    save_transcript(transcript_json_out, transcript)
    print(f"→ Saved diarized transcript to {transcript_json_out}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diarize_and_align.py WAV_PATH IN_JSON OUT_JSON")
        sys.exit(1)

    wav_path, transcript_in, transcript_out = sys.argv[1:]
    align_speakers(wav_path, transcript_in, transcript_out)

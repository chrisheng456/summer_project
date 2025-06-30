#!/usr/bin/env python3
import os
import json
import argparse
import tempfile
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline

# 加载模型
HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"

# 核心函数：说话人分离 + ASR 对齐
def diarize_and_transcribe(wav_path: str, output_json: str, hf_token: str):
    print("▶ Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=hf_token
    )
    diarization = pipeline(wav_path)

    print("▶ Loading Whisper ASR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("small", device=device)

    print("▶ Loading audio and slicing by speaker segments...")
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.to(device)

    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = segment.start, segment.end
        sf, ef = int(start * sample_rate), int(end * sample_rate)
        clip = waveform[:, sf:ef]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, clip.cpu(), sample_rate)
            result = asr_model.transcribe(tmp.name, fp16=(device=="cuda"))
        os.unlink(tmp.name)

        text = result.get("text", "").strip()
        segments.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end":   round(end, 3),
            "text":  text
        })

    # 排序并写入 JSON
    segments.sort(key=lambda x: x["start"])
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"✅ Aligned JSON saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step diarization + ASR: input WAV → aligned JSON in script directory"
    )
    parser.add_argument(
        "wav",
        help="输入 WAV 文件路径"
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv(HF_TOKEN_ENV),
        help=f"HuggingFace 访问令牌，或设置环境变量 {HF_TOKEN_ENV}"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        parser.error(f"找不到音频文件: {args.wav}")
    if not args.hf_token:
        parser.error(
            f"请通过 --hf_token 或环境变量 {HF_TOKEN_ENV} 提供 HuggingFace 访问令牌。"
        )

    # 自动生成输出文件名：脚本同目录，基于 wav 名称
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.splitext(os.path.basename(args.wav))[0] + "_aligned.json"
    output_path = os.path.join(script_dir, base)

    diarize_and_transcribe(args.wav, output_path, args.hf_token)
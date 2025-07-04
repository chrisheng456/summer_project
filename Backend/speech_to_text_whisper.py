#!/usr/bin/env python3
# speech_to_text_whisper.py
# 完全离线、基于 OpenAI Whisper 的语音转文本示例脚本
# 脚本内固定输入输出路径，输出 JSON 名称根据输入文件名自动生成

import os
import json
import whisper
from pathlib import Path

# 1. 直接在此处配置模型名和输入音频路径
MODEL_NAME   = "base"  # 可选 tiny, base, small, medium, large
INPUT_AUDIO  = "Trustee Meeting Recording (30 June 2025) V1.m4a"

def transcribe_whisper(model_name: str, audio_path: str):
    """
    使用 Whisper 模型对单个音频文件进行转录。
    返回一个包含若干段的列表，每段是 dict:
      {"start": float, "end": float, "text": str}
    """
    print(f"→ Loading Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    segments = result.get("segments", [])
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in segments
    ]

def save_transcript(transcript, out_json: str):
    """
    将转录结果保存为 JSON
    """
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"✅ Transcription saved to '{out_json}'")

def main():
    # 校验输入文件是否存在
    audio_path = Path(INPUT_AUDIO)
    if not audio_path.is_file():
        print(f"❌ 输入音频文件不存在：{audio_path}")
        return

    # 根据输入文件名自动生成输出 JSON 名称
    stem = audio_path.stem  # e.g. "Bdb001.interaction"
    # 如果 stem 包含多个点，只取第一个部分
    stem_base = stem.split(".")[0]
    output_json = audio_path.parent / f"{stem_base}_whisper.json"

    # 转录并保存
    transcript = transcribe_whisper(MODEL_NAME, str(audio_path))
    print(f"→ Transcribed {len(transcript)} segments")
    save_transcript(transcript, str(output_json))

if __name__ == "__main__":
    main()
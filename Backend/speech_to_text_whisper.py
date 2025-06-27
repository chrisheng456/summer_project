#!/usr/bin/env python3
# speech_to_text_whisper.py
# 完全离线、基于 OpenAI Whisper 的语音转文本示例脚本

import sys
import json
import whisper

def transcribe_whisper(model_name: str, audio_path: str):
    """
    使用 Whisper 模型对单个音频文件进行转录。
    返回一个包含若干段的列表，每段是 dict:
      {"start": float, "end": float, "text": str}
    """
    model = whisper.load_model(model_name)
    # 对 audio_path 做自动划分并转录
    result = model.transcribe(audio_path, fp16=False)
    segments = result.get("segments", [])
    # 只保留 start/end/text
    return [
        {
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"].strip()
        }
        for seg in segments
    ]

def save_transcript(transcript, out_json: str):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

def main():
    if len(sys.argv) != 4:
        print("Usage: python speech_to_text_whisper.py <model> <input_audio> <output_json>")
        print("Example: python speech_to_text_whisper.py base meeting.wav meeting_whisper.json")
        sys.exit(1)

    model_name   = sys.argv[1]   # e.g. "tiny", "base", "small", "medium", "large"
    audio_path   = sys.argv[2]
    output_json  = sys.argv[3]

    print(f"→ Loading Whisper model '{model_name}' …")
    transcript = transcribe_whisper(model_name, audio_path)
    print(f"→ Transcribed {len(transcript)} segments, saving to '{output_json}' …")
    save_transcript(transcript, output_json)
    print("✅ Whisper transcription done.")

if __name__ == "__main__":
    main()

"""
speech_to_text_01.py
------------------------------------
• TrueText（自动标点 + 首字母大写）
• 说话人分离（Diarization）
• 保存为 Word 和 JSON（仅包含 start/end/text）
• 用 ffmpeg 自动把 .m4a 转成 16kHz 单声道 WAV
"""
import os
import json
import threading
import subprocess
from datetime import datetime
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk  # pip install -U azure-cognitiveservices-speech
from docx import Document                         # pip install python-docx

# 1. 读取密钥
speech_key     = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION", "ukwest")
if not speech_key:
    raise RuntimeError("❌ 找不到 AZURE_SPEECH_KEY，请先在系统变量或 .env 中设置")

# 2. SpeechConfig & 功能开关
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
    "TrueText"
)
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
    "true"
)
speech_config.request_word_level_timestamps()

# 3. 用 ffmpeg 自动把 .m4a 转成 16kHz 单声道 WAV
BASE_DIR = Path(__file__).parent
src_path = BASE_DIR / "Trustee Meeting Recording (30 June 2025) V1.m4a"
wav_path = BASE_DIR / "tmp_converted.wav"

if not src_path.exists():
    raise FileNotFoundError(f"找不到源文件：{src_path.resolve()}")

print(f"🔄 转换音频：{src_path.name} → {wav_path.name} （16kHz 单声道 WAV）")
subprocess.run([
    "ffmpeg", "-y",
    "-i", str(src_path),
    "-ac", "1",
    "-ar", "16000",
    str(wav_path)
], check=True)

# 4. 准备 Azure 转写
audio_config = speechsdk.AudioConfig(filename=str(wav_path))
transcriber  = speechsdk.transcription.ConversationTranscriber(
    speech_config=speech_config,
    audio_config=audio_config
)

lines   = []             # 存储每句的时间戳和文本
is_done = threading.Event()

def _on_transcribed(evt: speechsdk.SpeechRecognitionEventArgs):
    if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
        return
    text = evt.result.text.strip()
    if not text:
        return
    start_sec = evt.result.offset / 10_000_000
    end_sec   = start_sec + (evt.result.duration / 10_000_000)
    lines.append({"start": start_sec, "end": end_sec, "text": text})
    print(f"[{start_sec:.2f}s - {end_sec:.2f}s] {text}")

def _on_session_stopped(_):
    print("=== 识别结束 ===")
    is_done.set()

def _on_canceled(evt):
    details = speechsdk.CancellationDetails(evt)
    print(f"CANCELED: {details.reason} / {details.error_details}")
    is_done.set()

transcriber.transcribed.connect(_on_transcribed)
transcriber.session_stopped.connect(_on_session_stopped)
transcriber.canceled.connect(_on_canceled)

# 5. 开始转写 & 等待结束
print(f"▶ 开始识别 {wav_path.name} ...")
transcriber.start_transcribing_async()
is_done.wait()
transcriber.stop_transcribing_async()

# 6. 整理 & 保存输出（不加时间戳，使用输入文件名前缀）
lines.sort(key=lambda x: x["start"])
base_name = src_path.stem  # 取音频文件名去掉扩展

docx_name = f"{base_name}.docx"
json_name = f"{base_name}.json"

# (a) 保存为 Word
doc = Document()
doc.add_heading("会议逐字稿", level=1)
for ln in lines:
    doc.add_paragraph(f"[{ln['start']:.2f}s - {ln['end']:.2f}s] {ln['text']}")
doc.save(docx_name)
print(f"✅ 已保存 Word：{docx_name}")

# (b) 保存为 JSON
with open(json_name, "w", encoding="utf-8") as f:
    json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 JSON：{json_name}")

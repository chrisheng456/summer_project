"""
speech_to_text.py
-----------------
• TrueText（自动标点 + 大小写）
• 实时说话人分离（Diarization）
• Word + JSON 输出
"""

import os
import json
import uuid
import time
import threading
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk   # pip install -U azure-cognitiveservices-speech
from docx import Document                            # pip install python-docx

# --------------------------------------------------
# 1. 读取密钥 —— 来自环境变量（或 .env，见注释）
# --------------------------------------------------
# 若你已经 `pip install python-dotenv` 并放了 .env，可以取消下一行注释：
# from dotenv import load_dotenv; load_dotenv()

speech_key     = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION", "eastus")

if not speech_key:
    raise RuntimeError("❌ 找不到 AZURE_SPEECH_KEY，请先在系统变量或 .env 中设置")

# --------------------------------------------------
# 2. SpeechConfig  &  特性开关
# --------------------------------------------------
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# ★ TrueText：自动标点 / 大小写 / 可选去填充词
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
    "TrueText"             # 或 "TrueText-NoFiller"
)

# ★ 说话人分离（Diarization）
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
    "true"
)

speech_config.request_word_level_timestamps()        # 字级时间戳（可选）

# --------------------------------------------------
# 3. 音频文件
# --------------------------------------------------
AUDIO_PATH = Path("test.wav")   # ← 改成你的文件
if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"找不到音频文件：{AUDIO_PATH.resolve()}")

audio_config = speechsdk.AudioConfig(filename=str(AUDIO_PATH))

# ConversationTranscriber 支持说话人分离
transcriber = speechsdk.transcription.ConversationTranscriber(
    speech_config=speech_config,
    audio_config=audio_config
)

# --------------------------------------------------
# 4. 事件回调
# --------------------------------------------------
lines = []                      # 每一句对象：speaker / text / offset / duration
done  = threading.Event()       # 用于阻塞主线程直到转写结束

def _on_transcribed(evt: speechsdk.SpeechRecognitionEventArgs):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        line = {
            "speaker"  : evt.result.speaker_id or "Unknown",
            "text"     : evt.result.text,
            "offset"   : evt.result.offset,    # 100-ns
            "duration" : evt.result.duration,
        }
        lines.append(line)
        print(f"[{line['speaker']}] {line['text']}")
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print("NOMATCH: 该片段无法识别")

def _on_session_stopped(_):
    print("=== 识别结束 ===")
    done.set()

def _on_canceled(evt):
    details = speechsdk.CancellationDetails(evt)
    print(f"CANCELED: {details.reason} / {details.error_details}")
    done.set()

transcriber.transcribed.connect(_on_transcribed)
transcriber.session_stopped.connect(_on_session_stopped)
transcriber.canceled.connect(_on_canceled)

# --------------------------------------------------
# 5. 开始转写并等待
# --------------------------------------------------
print(f"▶ 开始识别 {AUDIO_PATH} ...")
transcriber.start_transcribing_async()
done.wait()                          # 阻塞直到 SessionStopped / Canceled
transcriber.stop_transcribing_async()

# --------------------------------------------------
# 6. 保存结果
# --------------------------------------------------
# 按时间排序（保险起见）
lines.sort(key=lambda x: x["offset"])
plain_text = "\n".join(f"{ln['speaker']}: {ln['text']}" for ln in lines)

# (a) Word
doc = Document()
doc.add_heading("会议逐字稿", level=1)
for ln in lines:
    doc.add_paragraph(f"{ln['speaker']}: {ln['text']}")
doc.save("meeting_transcription.docx")
print("✅ 已保存 Word：meeting_transcription.docx")

# (b) JSON
output = {
    "id"              : str(uuid.uuid4()),
    "transcription"   : plain_text,
    "lines"           : lines,
    "abstract_summary": None,
    "key_points"      : [],
    "action_items"    : [],
    "sentiment"       : None,
    "attachment"      : [],
}
with open("meeting_minutes.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("✅ 已保存 JSON：meeting_minutes.json")

# check_azure_stt.py
from pathlib import Path
import os
import azure.cognitiveservices.speech as speechsdk

# 1) 用脚本所在目录定位 test.wav
BASE = Path(__file__).parent
AUDIO_PATH = BASE / "test.wav"     # 确保 Backend\test.wav 确实存在
print("audio path:", AUDIO_PATH)
print("exists   :", AUDIO_PATH.exists())

# 2) 读环境变量（.env 已加载到进程就行）
key    = os.getenv("AZURE_SPEECH_KEY")
region = os.getenv("AZURE_SPEECH_REGION")
print("using:", region, (key or "")[:5] + "...")

# 3) 走“文件识别”，强制连云端
cfg   = speechsdk.SpeechConfig(subscription=key, region=region)
audio = speechsdk.AudioConfig(filename=str(AUDIO_PATH))
reco  = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

res = reco.recognize_once_async().get()
print("reason:", res.reason)

if res.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("text:", res.text)
elif res.reason == speechsdk.ResultReason.NoMatch:
    print("no match (认证 OK，但音频里没识别到文字)")
elif res.reason == speechsdk.ResultReason.Canceled:
    details = speechsdk.CancellationDetails(res)
    print("cancel reason :", details.reason)
    print("error code    :", getattr(details, "error_code", None))
    print("error details :", getattr(details, "error_details", None))

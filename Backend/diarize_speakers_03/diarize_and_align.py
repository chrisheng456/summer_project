import os
import time
import json
import glob
import subprocess
import requests
from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions
)
from datetime import datetime, timedelta

# === 加载环境变量 ===
load_dotenv()
speech_key = os.getenv("AZURE_SPEECH_KEY")
region = os.getenv("AZURE_SPEECH_REGION", "uksouth")
storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER", "cunchu")
if not all([speech_key, region, storage_conn]):
    raise ValueError("请设置 AZURE_SPEECH_KEY, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER")

# === 本地配置 ===
orig = "Trustee Meeting Recording (30 June 2025) V1.m4a"
denoised = "denoised.wav"
segment_pattern = "seg_%03d.wav"
output_file = "result.json"
locale = "en-US"
segment_duration = 300  # 每段 300 秒

# --- 1. 降噪 & 转 WAV ---
print("🎵 降噪并转为 WAV ...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", orig,
    "-ar", "16000",
    "-ac", "1",
    "-af", "afftdn",     # FFT 降噪
    "-acodec", "pcm_s16le",
    denoised
], check=True)

# --- 2. 分段 ---
print("⏳ 按 5 分钟切分音频 ...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", denoised,
    "-f", "segment",
    "-segment_time", str(segment_duration),
    "-c", "copy",
    segment_pattern
], check=True)

# Azure Blob 客户端
blob_svc = BlobServiceClient.from_connection_string(storage_conn)
container = blob_svc.get_container_client(container_name)

all_lines = []

for idx, wav in enumerate(sorted(glob.glob("seg_*.wav"))):
    offset_base = idx * segment_duration
    blob_name = os.path.basename(wav)
    print(f"\n📤 上传并转录段 {idx}: {wav} → {blob_name}")

    # 上传
    blob = container.get_blob_client(blob_name)
    with open(wav, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    sas = generate_blob_sas(
        account_name=blob_svc.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_svc.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=2)
    )
    url = f"https://{blob_svc.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas}"

    # 创建转录任务
    endpoint = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json"
    }
    body = {
        "displayName": f"Batch Transcription Segment {idx}",
        "locale": locale,
        "contentUrls": [url],
        "properties": {
            "diarizationEnabled": True,
            "diarization": {
                "speakers": {"minCount": 2, "maxCount": 8}
            },
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    resp = requests.post(endpoint, headers=headers, json=body)
    resp.raise_for_status()
    trans_url = resp.headers["Location"]

    # 轮询
    print("   等待转录完成...", end="", flush=True)
    while True:
        st = requests.get(trans_url, headers=headers).json()["status"]
        print(".", end="", flush=True)
        if st in ("Succeeded","Failed"):
            break
        time.sleep(10)
    print(f" {st}")

    if st == "Failed":
        print("   ⚠️ 段转录失败，跳过")
        continue

    # 拉取结果
    files_url = requests.get(trans_url, headers=headers).json()["links"]["files"]
    files = requests.get(files_url, headers=headers).json()["values"]
    tf = next(f for f in files if f["kind"]=="Transcription")
    data = requests.get(tf["links"]["contentUrl"]).json()

    # 提取 recognizedPhrases
    for phrase in data.get("recognizedPhrases", []):
        if not phrase.get("nBest"):
            continue
        best = phrase["nBest"][0]
        start = phrase.get("offsetInTicks",0)/1e7 + offset_base
        dur   = phrase.get("durationInTicks",0)/1e7
        speaker = f"SPEAKER_{phrase.get('speaker',0):02d}"
        all_lines.append({
            "start": round(start,2),
            "end":   round(start+dur,2),
            "text":  best.get("display",""),
            "speaker": speaker
        })

# --- 保存合并结果 ---
with open(output_file,"w",encoding="utf-8") as f:
    json.dump({"lines": all_lines}, f, ensure_ascii=False, indent=2)

print(f"\n🎉 全部完成，已保存到 {output_file}")
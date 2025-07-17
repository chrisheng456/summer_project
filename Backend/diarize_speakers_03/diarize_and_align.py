import os
import time
import json
import requests
import subprocess
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# === 加载环境变量 ===
load_dotenv()

# === 环境变量读取 ===
speech_key = os.getenv("AZURE_SPEECH_KEY")
region = os.getenv("AZURE_SPEECH_REGION", "uksouth")
storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER", "cunchu")

if not all([speech_key, region, storage_connection_string]):
    raise ValueError("请设置 AZURE_SPEECH_KEY, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER 等环境变量")

# === 本地配置 ===
original_audio_path = "Trustee Meeting Recording (30 June 2025) V1.m4a"
converted_audio_path = "output.wav"
output_file = "result.json"
locale = "en-US"

# === Step 1: 音频格式转换 ===
if not original_audio_path.lower().endswith(".wav"):
    print(f"🎵 转换音频为 WAV 格式: {converted_audio_path}")
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", original_audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        converted_audio_path
    ]
    subprocess.run(ffmpeg_command, check=True)
    local_file_path = converted_audio_path
    print("✅ 音频转换完成")
else:
    local_file_path = original_audio_path
    print("🎧 输入已为 WAV 格式，直接使用")

# === 上传到 Blob Storage ===
blob_name = os.path.basename(local_file_path).replace(" ", "_")
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

print(f"Uploading {local_file_path} to Azure Blob as {blob_name}...")
with open(local_file_path, "rb") as f:
    blob_client.upload_blob(f, overwrite=True)

sas_token = generate_blob_sas(
    account_name=blob_service_client.account_name,
    container_name=container_name,
    blob_name=blob_name,
    account_key=blob_service_client.credential.account_key,
    permission=BlobSasPermissions(read=True),
    expiry=datetime.utcnow() + timedelta(hours=2)
)

blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
print("✅ Blob SAS URL generated.")

# === 创建转录任务 ===
endpoint = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
headers = {"Ocp-Apim-Subscription-Key": speech_key, "Content-Type": "application/json"}
body = {
    "displayName": "Batch Transcription Job",
    "locale": locale,
    "contentUrls": [blob_url],
    "properties": {
        "diarizationEnabled": True,
        "wordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked"
    }
}

print("Submitting transcription job...")
response = requests.post(endpoint, headers=headers, json=body)
response.raise_for_status()
transcription_url = response.headers["Location"]
print("✅ Job submitted. Transcription URL:", transcription_url)

# === 轮询任务状态 ===
print("Waiting for transcription job to complete...")
while True:
    status_resp = requests.get(transcription_url, headers=headers)
    status_resp.raise_for_status()
    status_data = status_resp.json()
    status = status_data["status"]
    print("⏳ Status:", status)
    if status in ["Succeeded", "Failed"]:
        break
    time.sleep(120)

if status == "Failed":
    print("❌ Transcription failed.")
    print(json.dumps(status_data, indent=2))
    exit(1)

# === 获取结果 JSON ===
files_url = status_data["links"]["files"]
files_resp = requests.get(files_url, headers=headers)
files_data = files_resp.json()

transcription_file = None
for file in files_data["values"]:
    if file["kind"] == "Transcription":
        transcription_file = file
        break

if not transcription_file:
    print("❌ No transcription result file found.")
    print(json.dumps(files_data, indent=2))
    exit(1)

results_url = transcription_file["links"]["contentUrl"]
print("Downloading transcription result from:", results_url)
results_resp = requests.get(results_url)
results_data = results_resp.json()

# === 保存原始 JSON 以调试 ===
with open("raw_response.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
print("✅ Saved raw API response to raw_response.json")

# === 提取结构化信息 ===
final_output = {"lines": []}
print("\n🔍 Analyzing API response structure...")
print(f"Response keys: {list(results_data.keys())}")

def parse_pt_time(pt_string):
    try:
        return float(pt_string.replace("PT", "").replace("S", ""))
    except:
        return 0.0

if "recognizedPhrases" in results_data:
    for phrase in results_data["recognizedPhrases"]:
        if "nBest" not in phrase or len(phrase["nBest"]) == 0:
            continue
        best = phrase["nBest"][0]
        speaker = f"SPEAKER_{phrase.get('speaker', 0):02d}"

        if "offset" in phrase:
            if isinstance(phrase["offset"], str):
                offset = parse_pt_time(phrase["offset"])
                duration = parse_pt_time(phrase["duration"])
            else:
                offset = phrase.get("offsetInTicks", 0) / 10**7
                duration = phrase.get("durationInTicks", 0) / 10**7
            start = round(offset, 2)
            end = round(offset + duration, 2)
        else:
            start, end = 0.0, 0.0

        final_output["lines"].append({
            "start": start,
            "end": end,
            "text": best["display"],
            "speaker": speaker
        })

elif "combinedRecognizedPhrases" in results_data:
    for segment in results_data["combinedRecognizedPhrases"]:
        for best in segment.get("nBest", []):
            speaker = f"SPEAKER_{segment.get('speaker', 0):02d}"
            text = best.get("display", "")
            final_output["lines"].append({
                "start": 0.0,
                "end": 0.0,
                "text": text,
                "speaker": speaker
            })

# 如果完全没有内容，尝试兜底策略
if not final_output["lines"]:
    if "display" in results_data:
        final_output["lines"].append({
            "start": 0.0,
            "end": 0.0,
            "text": results_data["display"],
            "speaker": "SPEAKER_00"
        })

# === 保存最终输出 ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"lines": final_output["lines"]}, f, indent=2)

print(f"\n🎉 Transcription complete. Output saved to: {output_file}")
print(f"📊 Total lines extracted: {len(final_output['lines'])}")
if not final_output["lines"]:
    print("❌ WARNING: No lines were extracted from the transcription results")
    print("Please check the 'raw_response.json' file for the full API response")
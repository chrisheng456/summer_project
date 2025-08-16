#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量上传 + 批处理转写（Azure Speech v3.1）→ 输出 {"lines":[...]}
- 所有中间文件（denoised.wav / seg_*.wav）写入系统临时目录并在结束时自动删除
- 一次性提交 contentUrls = [每段的只读SAS URL]（Azure 后台并行）
- 遍历所有结果文件 → 统一拼接成 {start,end,text,speaker} 的 lines

必需环境变量：
  AZURE_SPEECH_KEY
  AZURE_REGION
  AZURE_STORAGE_CONNECTION_STRING
  AZURE_STORAGE_CONTAINER
"""

import os
import re
import json
import time
import uuid
import glob
import logging
import subprocess
import tempfile
from urllib.parse import urlparse, urlunparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# ---------- 基本设置 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SPEECH_KEY   = os.environ["AZURE_SPEECH_KEY"]
REGION       = os.environ["AZURE_REGION"]
CONN_STR     = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER    = os.environ["AZURE_STORAGE_CONTAINER"]

INPUT_FILE   = os.getenv("INPUT_FILE", "Trustee Meeting Recording (30 June 2025) V1.m4a")
SEG_SECONDS  = int(os.getenv("SEGMENT_SECONDS", "180"))  # 默认3分钟
LOCALE       = os.getenv("LOCALE", "en-US")
MAX_WORKERS  = int(os.getenv("MAX_UPLOAD_WORKERS", "8"))

OUTPUT_FILE  = "result.json"

SESSION = requests.Session()
SESSION.headers.update({
    "Ocp-Apim-Subscription-Key": SPEECH_KEY,
    "Content-Type": "application/json"
})
TRANSCRIPTIONS_ENDPOINT = f"https://{REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"

# ---------- 小工具 ----------
def run_ffmpeg(cmd, title):
    logging.info(title)
    subprocess.run(cmd, check=True)

def ticks_to_seconds(x):
    if x is None:
        return 0.0
    try:
        return float(x) / 10_000_000.0
    except Exception:
        try:
            return int(x) / 10_000_000.0
        except Exception:
            return 0.0

def strip_query(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def norm_speaker(spk):
    if spk is None:
        return "SPEAKER_00"
    if isinstance(spk, str) and spk.startswith("SPEAKER_"):
        return spk
    try:
        num = int(spk)
        return f"SPEAKER_{num:02d}"
    except Exception:
        return "SPEAKER_00"

# ---------- Blob ----------
blob_svc = BlobServiceClient.from_connection_string(CONN_STR)
container_client = blob_svc.get_container_client(CONTAINER)
try:
    container_client.create_container()
except Exception:
    pass

ACCOUNT_NAME = blob_svc.account_name
try:
    ACCOUNT_KEY = blob_svc.credential.account_key
except Exception:
    ACCOUNT_KEY = None
if not ACCOUNT_KEY:
    raise RuntimeError("连接串中未解析出 AccountKey，请检查 AZURE_STORAGE_CONNECTION_STRING。")

BLOB_BASE = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER}"

def upload_with_read_sas(local_path: str, blob_name: str, hours=6) -> str:
    blob = container_client.get_blob_client(blob_name)
    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    sas = generate_blob_sas(
        account_name=ACCOUNT_NAME,
        container_name=CONTAINER,
        blob_name=blob_name,
        account_key=ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=hours),
    )
    return f"{BLOB_BASE}/{blob_name}?{sas}"

# ---------- 主流程 ----------
def main():
    with tempfile.TemporaryDirectory(prefix="asr_tmp_") as tmpdir:
        denoised = os.path.join(tmpdir, "denoised.wav")
        seg_pattern = os.path.join(tmpdir, "seg_%03d.wav")

        # 1) 转 WAV（16kHz/mono），输出到临时目录
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", INPUT_FILE,
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            denoised
        ], "🎵 降噪并转为 WAV ...")

        # 2) 按固定长度切分到临时目录
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", denoised,
            "-f", "segment",
            "-segment_time", str(SEG_SECONDS),
            "-c", "copy",
            seg_pattern
        ], f"⏳ 按 {SEG_SECONDS//60} 分钟切分音频 ...")

        segments = sorted(glob.glob(os.path.join(tmpdir, "seg_*.wav")))
        logging.info(f"✅ 切分完成，共 {len(segments)} 段")

        # 3) 并行上传 + 生成只读 SAS
        logging.info("📤 并行上传所有分段到 Azure Blob（并生成只读 SAS） ...")

        def _one(seg_path):
            blob_name = f"{uuid.uuid4().hex}_{os.path.basename(seg_path)}"
            sas_url = upload_with_read_sas(seg_path, blob_name)
            return seg_path, blob_name, sas_url

        uploaded = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_one, p) for p in segments]
            for fut in as_completed(futs):
                uploaded.append(fut.result())

        uploaded.sort(key=lambda x: x[0])
        content_urls = [x[2] for x in uploaded]
        src_base_to_index = {strip_query(url): i for i, url in enumerate(content_urls)}

        logging.info(f"✅ 已上传 {len(content_urls)} 段音频")

        # 4) 提交批处理任务
        body = {
            "displayName": "Batch meeting transcription",
            "locale": LOCALE,
            "contentUrls": content_urls,
            "properties": {
                "diarizationEnabled": True,
                "diarization": {"speakers": {"minCount": 2, "maxCount": 8}},
                "wordLevelTimestampsEnabled": True,
                "punctuationMode": "DictatedAndAutomatic",
                "profanityFilterMode": "Masked"
            }
        }
        r = SESSION.post(TRANSCRIPTIONS_ENDPOINT, json=body, timeout=60)
        r.raise_for_status()
        trans_url = r.headers.get("Location") or r.json().get("self")
        logging.info(f"📌 任务 URL: {trans_url}")

        # 5) 轮询状态（60s）
        while True:
            time.sleep(60)
            st = SESSION.get(trans_url, timeout=60).json()
            status = st.get("status")
            logging.info(f"状态: {status}")
            if status in ("Succeeded", "Failed"):
                if status != "Succeeded":
                    raise RuntimeError("转录失败")
                break

        # 6) 遍历所有 transcription 文件
        files_url = st.get("links", {}).get("files")
        files_list = SESSION.get(files_url, timeout=60).json().get("values", [])
        trans_files = [f for f in files_list if f.get("kind") == "Transcription"]
        if not trans_files:
            raise RuntimeError("未找到结果文件")

        lines = []
        for tf in trans_files:
            raw = SESSION.get(tf["links"]["contentUrl"], timeout=120).json()
            src_url = raw.get("source")
            seg_idx = src_base_to_index.get(strip_query(src_url)) if src_url else None

            for p in raw.get("recognizedPhrases", []):
                nbest = p.get("nBest") or []
                if not nbest:
                    continue
                text = nbest[0].get("display") or ""

                # 时间戳
                if "offsetMilliseconds" in p and "durationMilliseconds" in p:
                    start_s = float(p["offsetMilliseconds"]) / 1000.0
                    dur_s   = float(p["durationMilliseconds"]) / 1000.0
                else:
                    start_s = ticks_to_seconds(p.get("offsetInTicks"))
                    dur_s   = ticks_to_seconds(p.get("durationInTicks"))

                # 如果时间小且有分段信息 → 认为是相对时间 → 加段偏移
                if src_url and start_s < SEG_SECONDS + 5:
                    base = (seg_idx or 0) * SEG_SECONDS
                    start_s += base

                lines.append({
                    "start": round(start_s, 2),
                    "end": round(start_s + dur_s, 2),
                    "text": text,
                    "speaker": norm_speaker(p.get("speaker") or p.get("speakerId")),
                })

        # 7) 输出
        lines.sort(key=lambda x: (x["start"], x["end"]))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)

        logging.info(f"🎉 完成：共 {len(lines)} 条，已保存 {OUTPUT_FILE}")
        # 离开 with 块时，tmpdir 会被自动删除

if __name__ == "__main__":
    main()
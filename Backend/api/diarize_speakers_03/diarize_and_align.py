#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
并行版 Azure 批量转写脚本（7 线程）
- 降噪 & 转 WAV（ffmpeg）
- 分段（每段 5 分钟，默认 300 秒）
- 并行(7) 上传到 Azure Blob 并创建转录任务
- 并行轮询任务状态，拉取结果
- 合并为 result.json

需要的环境变量：
  AZURE_SPEECH_KEY
  AZURE_SPEECH_REGION            # 例如 uksouth
  AZURE_STORAGE_CONNECTION_STRING
  AZURE_STORAGE_CONTAINER        # 例如 cunchu

可选环境变量：
  AZURE_LOCALE                   # 默认为 en-US
  SEGMENT_SECONDS                # 默认 300
  INPUT_FILE                     # 默认 "Trustee Meeting Recording (30 June 2025) V1.m4a"
  CLEANUP_BLOBS                  # 结果下载后是否删除 blob，"1" 表示删除
"""

import os
import time
import json
import glob
import uuid
import math
import shutil
import random
import logging
import subprocess
import concurrent.futures
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)

# --------------------- 基础设置 ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

load_dotenv(override=True)

SPEECH_KEY   = os.getenv("AZURE_SPEECH_KEY")
REGION       = os.getenv("AZURE_SPEECH_REGION", "uksouth")
CONN_STR     = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER    = os.getenv("AZURE_STORAGE_CONTAINER", "cunchu")
LOCALE       = os.getenv("AZURE_LOCALE", "en-US")

SEGMENT_SEC  = int(os.getenv("SEGMENT_SECONDS", "300"))
INPUT_FILE   = os.getenv("INPUT_FILE", "Trustee Meeting Recording (30 June 2025) V1.m4a")
DENOISED     = "denoised.wav"
SEG_PATTERN  = "seg_%03d.wav"
OUTPUT_FILE  = "result.json"
CLEANUP_BLOBS= os.getenv("CLEANUP_BLOBS", "0") == "1"

MAX_WORKERS  = 7  # ✅ 并发线程数（按你要求设置成 7）

if not all([SPEECH_KEY, REGION, CONN_STR, CONTAINER]):
    raise ValueError("请设置 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION / AZURE_STORAGE_CONNECTION_STRING / AZURE_STORAGE_CONTAINER")

# requests 会话（复用连接）
SESSION = requests.Session()
SESSION.headers.update({
    "Ocp-Apim-Subscription-Key": SPEECH_KEY
})

TRANSCRIPTIONS_ENDPOINT = f"https://{REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"


# --------------------- 小工具函数 ---------------------
def run_cmd(cmd: list, desc: str):
    logging.info(desc + " ...")
    logging.debug("CMD: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ticks_to_seconds(x):
    """Azure 批量转写返回的 offsetInTicks/durationInTicks 可能是 str 或 int"""
    if x is None:
        return 0.0
    try:
        return int(x) / 10_000_000.0  # 1 tick = 100ns
    except Exception:
        try:
            return float(x) / 10_000_000.0
        except Exception:
            return 0.0


def backoff_sleep(base=2.0, attempt=0, jitter=True, max_sleep=30.0):
    """指数退避 + 抖动"""
    s = min(max_sleep, (base ** attempt))
    if jitter:
        s *= (0.5 + random.random() * 0.5)
    time.sleep(s)


# --------------------- Azure 相关 ---------------------
blob_svc = BlobServiceClient.from_connection_string(CONN_STR)
container_client = blob_svc.get_container_client(CONTAINER)

# 账号名 / Key（从连接串里解析）
ACCOUNT_NAME = blob_svc.account_name
try:
    ACCOUNT_KEY = blob_svc.credential.account_key  # 只有使用连接串才有
except Exception:
    ACCOUNT_KEY = None

if not ACCOUNT_KEY:
    raise RuntimeError("无法从连接串中解析出 account key，请确认 AZURE_STORAGE_CONNECTION_STRING 正确且包含 AccountKey。")


def upload_blob_and_get_sas(local_path: str, blob_name: str, expiry_hours=3) -> str:
    """上传本地文件到 Blob，并返回带 SAS 的可读 URL"""
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    sas = generate_blob_sas(
        account_name=ACCOUNT_NAME,
        container_name=CONTAINER,
        blob_name=blob_name,
        account_key=ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
    )
    url = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER}/{blob_name}?{sas}"
    return url


def create_transcription(url: str, display_name: str) -> str:
    """创建批量转写任务，返回 trans_url（轮询用）"""
    body = {
        "displayName": display_name,
        "locale": LOCALE,
        "contentUrls": [url],
        "properties": {
            "diarizationEnabled": True,
            "diarization": {"speakers": {"minCount": 2, "maxCount": 8}},
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = SESSION.post(TRANSCRIPTIONS_ENDPOINT, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.headers["Location"]  # 任务状态查询 URL


def poll_until_done(trans_url: str, timeout_minutes=120, poll_seconds=8) -> dict:
    """轮询任务直到结束，返回最终任务 JSON；如果失败抛异常"""
    t0 = time.time()
    while True:
        r = SESSION.get(trans_url, timeout=20)
        r.raise_for_status()
        data = r.json()
        status = str(data.get("status"))
        if status in ("Succeeded", "Failed"):
            if status == "Failed":
                raise RuntimeError(f"转写任务失败：{data}")
            return data
        if (time.time() - t0) > timeout_minutes * 60:
            raise TimeoutError("转写轮询超时")
        time.sleep(poll_seconds)


def fetch_transcription_text(task_json: dict) -> list[dict]:
    """从任务 JSON 里找到转写文件，下载内容并解析为行列表"""
    links = task_json.get("links", {}) or {}
    files_url = links.get("files")
    if not files_url:
        return []

    fr = SESSION.get(files_url, timeout=30)
    fr.raise_for_status()
    values = (fr.json() or {}).get("values", [])

    tf = None
    for v in values:
        if v.get("kind") == "Transcription":
            tf = v
            break
    if not tf:
        return []

    content_url = tf["links"]["contentUrl"]
    cr = SESSION.get(content_url, timeout=60)
    cr.raise_for_status()
    data = cr.json()

    lines = []
    for phrase in data.get("recognizedPhrases", []) or []:
        nbest = phrase.get("nBest") or []
        if not nbest:
            continue
        best = nbest[0]
        start = ticks_to_seconds(phrase.get("offsetInTicks"))
        dur   = ticks_to_seconds(phrase.get("durationInTicks"))
        speaker_id = phrase.get("speaker", 0)
        lines.append({
            "start": start,
            "end":   start + dur,
            "text":  best.get("display", ""),
            "speaker": f"SPEAKER_{int(speaker_id):02d}"
        })
    return lines


# --------------------- 并行任务（每段） ---------------------
def process_one_segment(idx: int, wav_path: str, segment_len: int, cleanup_blob=False) -> list[dict]:
    """处理单个分段：上传 → 创建任务 → 轮询 → 拉结果；返回带 offset 校正后的行"""
    offset_base = idx * segment_len
    blob_name = f"{uuid.uuid4().hex}_{os.path.basename(wav_path)}"
    display = f"Batch Transcription Segment {idx:03d}"

    # 带重试的上传+创建+轮询+下载
    attempts = 0
    while True:
        attempts += 1
        try:
            logging.info(f"[{idx:03d}] 上传并创建任务: {wav_path}")
            url = upload_blob_and_get_sas(wav_path, blob_name)

            trans_url = create_transcription(url, display)
            logging.info(f"[{idx:03d}] 任务已创建，开始轮询 ...")

            task_json = poll_until_done(trans_url)
            lines = fetch_transcription_text(task_json)

            # 校正 time offset
            for ln in lines:
                ln["start"] = round(ln["start"] + offset_base, 2)
                ln["end"]   = round(ln["end"]   + offset_base, 2)

            logging.info(f"[{idx:03d}] 完成，提取到 {len(lines)} 条")
            return lines

        except Exception as e:
            if attempts >= 5:
                logging.error(f"[{idx:03d}] 失败（已重试 {attempts} 次）：{e}")
                return []
            logging.warning(f"[{idx:03d}] 出错（第 {attempts} 次）：{e}，退避后重试")
            backoff_sleep(attempt=attempts)

        finally:
            if cleanup_blob:
                try:
                    container_client.delete_blob(blob_name)
                except Exception:
                    pass


# --------------------- 主流程 ---------------------
def main():
    # 1) 降噪并转 WAV
    run_cmd([
        "ffmpeg", "-y",
        "-i", INPUT_FILE,
        "-ar", "16000",
        "-ac", "1",
        "-af", "afftdn",
        "-acodec", "pcm_s16le",
        DENOISED
    ], "🎵 降噪并转为 WAV")

    # 2) 切分（每段 SEGMENT_SEC 秒）
    run_cmd([
        "ffmpeg", "-y",
        "-i", DENOISED,
        "-f", "segment",
        "-segment_time", str(SEGMENT_SEC),
        "-c", "copy",
        SEG_PATTERN
    ], "⏳ 按固定长度切分音频")

    segments = sorted(glob.glob("seg_*.wav"))
    if not segments:
        logging.warning("未生成任何分段文件，退出")
        return

    logging.info(f"共 {len(segments)} 段，将并发处理（max_workers={MAX_WORKERS}）")

    all_lines: list[dict] = []

    # 3) 并行上传/创建任务/轮询/拉结果
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [
            ex.submit(process_one_segment, idx, wav, SEGMENT_SEC, CLEANUP_BLOBS)
            for idx, wav in enumerate(segments)
        ]
        for fut in concurrent.futures.as_completed(futs):
            lines = fut.result() or []
            all_lines.extend(lines)

    # 4) 合并 & 排序 & 保存
    all_lines.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"lines": all_lines}, f, ensure_ascii=False, indent=2)

    logging.info(f"🎉 全部完成，共 {len(all_lines)} 条，已保存到 {OUTPUT_FILE}")

    # （可选）清理中间文件
    # shutil.rmtree(...) 或删除 seg_*.wav/denoised.wav
    # 视需求保留或清理
    # for p in segments + [DENOISED]:
    #     try:
    #         os.remove(p)
    #     except Exception:
    #         pass


if __name__ == "__main__":
    main()
from __future__ import annotations
import os, json, glob, uuid, time, tempfile, subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# ---------------- helpers ----------------
def _ticks_to_seconds(x) -> float:
    if x is None: return 0.0
    try: return float(x) / 10_000_000.0
    except Exception:
        try: return int(x) / 10_000_000.0
        except Exception: return 0.0

def _strip_query(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def _norm_speaker(spk) -> str:
    if spk is None: return "SPEAKER_00"
    if isinstance(spk, str) and spk.startswith("SPEAKER_"): return spk
    try: return f"SPEAKER_{int(spk):02d}"
    except Exception: return "SPEAKER_00"

def _run_ffmpeg(cmd: list[str]):
    subprocess.run(cmd, check=True)

# ---------------- core ----------------
def azure_batch_diarize(
    input_wav_path: str,
    *,
    segment_seconds: int = 300,
    locale: str = "en-US",
    parallel_jobs: bool = True,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Azure Speech v3.1 批量转写 + 说话人分离。
    - 并行上传分片
    - parallel_jobs=True: 为每个分片创建单独 Job，并发轮询与取结果（真正并行）
      False: 把所有 contentUrls 放进一个 Job（等整体完成后统一取结果）
    返回: [{"start": float, "end": float, "text": str, "speaker": "SPEAKER_00"}, ...]
    需要环境变量：
      AZURE_SPEECH_KEY
      AZURE_SPEECH_REGION  (或 AZURE_REGION)
      AZURE_STORAGE_CONNECTION_STRING
      AZURE_STORAGE_CONTAINER
    """
    speech_key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("AZURE_REGION")
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_STORAGE_CONTAINER")
    if not (speech_key and region and conn_str and container):
        raise RuntimeError("缺少 Azure 批量所需环境变量（SPEECH/REGION/STORAGE/CONTAINER）")

    # Blob client
    blob_svc = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_svc.get_container_client(container)
    try: container_client.create_container()
    except Exception: pass
    account_name = blob_svc.account_name
    try: account_key = blob_svc.credential.account_key  # type: ignore[attr-defined]
    except Exception: account_key = None
    if not account_key:
        raise RuntimeError("无法从连接串解析 AccountKey")
    blob_base = f"https://{account_name}.blob.core.windows.net/{container}"

    def _upload_with_read_sas(local_path: str, blob_name: str, hours=6) -> str:
        blob = container_client.get_blob_client(blob_name)
        with open(local_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=hours),
        )
        return f"{blob_base}/{blob_name}?{sas}"

    session = requests.Session()
    session.headers.update({
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json",
    })
    transcriptions_endpoint = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"

    with tempfile.TemporaryDirectory(prefix="asr_tmp_") as tmpdir:
        # 1) 规范化音频
        wav_16k = os.path.join(tmpdir, "audio_16k.wav")
        _run_ffmpeg(["ffmpeg","-y","-loglevel","error","-i",input_wav_path,"-ar","16000","-ac","1","-acodec","pcm_s16le",wav_16k])

        # 2) 切段
        seg_pattern = os.path.join(tmpdir, "seg_%03d.wav")
        _run_ffmpeg(["ffmpeg","-y","-loglevel","error","-i",wav_16k,"-f","segment","-segment_time",str(segment_seconds),"-c","copy",seg_pattern])
        segments = sorted(glob.glob(os.path.join(tmpdir, "seg_*.wav")))
        if not segments: return []

        # 3) 并行上传
        urls: List[str] = [None] * len(segments)  # type: ignore
        with ThreadPoolExecutor(max_workers=min(max_workers, len(segments))) as ex:
            futs = {ex.submit(_upload_with_read_sas, seg, f"{uuid.uuid4().hex}_{os.path.basename(seg)}"): i for i, seg in enumerate(segments)}
            for fut in as_completed(futs):
                i = futs[fut]
                urls[i] = fut.result()

        # 4) 创建 job（单 job 或多 job）
        def _create_job(content_urls: List[str]) -> str:
            body = {
                "displayName": "Batch diarization",
                "locale": locale,
                "contentUrls": content_urls,
                "properties": {
                    "diarizationEnabled": True,
                    "diarization": {"speakers": {"minCount": 2, "maxCount": 8}},
                    "wordLevelTimestampsEnabled": True,
                    "punctuationMode": "DictatedAndAutomatic",
                    "profanityFilterMode": "Masked",
                },
            }
            r = session.post(transcriptions_endpoint, json=body, timeout=60)
            r.raise_for_status()
            return r.headers.get("Location") or r.json().get("self")

        def _poll_and_fetch(trans_url: str) -> Dict[str, Any]:
            # 轮询
            while True:
                time.sleep(20)
                st = session.get(trans_url, timeout=60).json()
                status = st.get("status")
                if status in ("Succeeded","Failed"):
                    if status != "Succeeded":
                        raise RuntimeError("Azure batch job failed")
                    files_url = st.get("links", {}).get("files")
                    break
            # files
            files_list = session.get(files_url, timeout=60).json().get("values", [])
            trans_files = [f for f in files_list if f.get("kind") == "Transcription"]
            # 多数情况下一个 job 只有一个 Transcription 文件
            results = []
            for tf in trans_files:
                results.append(session.get(tf["links"]["contentUrl"], timeout=120).json())
            return {"files": results}

        lines: List[Dict[str, Any]] = []

        if parallel_jobs:
            # 多 Job 并行：每个分片一个 Job
            with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
                # 创建所有 job
                job_urls = []
                for u in urls:
                    job_urls.append(_create_job([u]))
                # 并行轮询
                futs = {ex.submit(_poll_and_fetch, ju): idx for idx, ju in enumerate(job_urls)}
                for fut in as_completed(futs):
                    seg_idx = futs[fut]
                    result = fut.result()
                    # 解析 recognizedPhrases
                    for raw in result["files"]:
                        for p in raw.get("recognizedPhrases", []):
                            nbest = p.get("nBest") or []
                            if not nbest: continue
                            text = (nbest[0].get("display") or "").strip()
                            if "offsetMilliseconds" in p and "durationMilliseconds" in p:
                                start_s = float(p["offsetMilliseconds"]) / 1000.0
                                dur_s   = float(p["durationMilliseconds"]) / 1000.0
                            else:
                                start_s = _ticks_to_seconds(p.get("offsetInTicks"))
                                dur_s   = _ticks_to_seconds(p.get("durationInTicks"))
                            # 分片时间偏移
                            base = seg_idx * segment_seconds
                            start_s += base
                            lines.append({
                                "start": round(start_s, 2),
                                "end":   round(start_s + dur_s, 2),
                                "text":  text,
                                "speaker": _norm_speaker(p.get("speaker") or p.get("speakerId")),
                            })
        else:
            # 单 Job：一次性丢所有 contentUrls（更省配额，但返回要等整体完成）
            trans_url = _create_job(urls)
            result = _poll_and_fetch(trans_url)
            # 建映射：源URL(无Query) -> 分片index
            idx_by_src = { _strip_query(u): i for i, u in enumerate(urls) }
            for raw in result["files"]:
                src_url = raw.get("source")
                seg_idx = idx_by_src.get(_strip_query(src_url)) if src_url else 0
                for p in raw.get("recognizedPhrases", []):
                    nbest = p.get("nBest") or []
                    if not nbest: continue
                    text = (nbest[0].get("display") or "").strip()
                    if "offsetMilliseconds" in p and "durationMilliseconds" in p:
                        start_s = float(p["offsetMilliseconds"]) / 1000.0
                        dur_s   = float(p["durationMilliseconds"]) / 1000.0
                    else:
                        start_s = _ticks_to_seconds(p.get("offsetInTicks"))
                        dur_s   = _ticks_to_seconds(p.get("durationInTicks"))
                    base = seg_idx * segment_seconds
                    start_s += base
                    lines.append({
                        "start": round(start_s, 2),
                        "end":   round(start_s + dur_s, 2),
                        "text":  text,
                        "speaker": _norm_speaker(p.get("speaker") or p.get("speakerId")),
                    })

        lines.sort(key=lambda x: (x["start"], x["end"]))
        return lines

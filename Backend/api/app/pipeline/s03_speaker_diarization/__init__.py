# Backend/api/app/pipeline/s03_speaker_diarization/__init__.py

from __future__ import annotations
import json, time, uuid,os
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
import httpx

from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
)

from ...config import app_config
from ...schema.process_information import ProcessInformation
from ...utils.azure_batch_diarizer import azure_batch_diarize

# 默认开启 Azure Batch diarization；如需强制关闭，可临时把这个常量改成 False
USE_AZURE_BATCH: bool = True
# --- 可配置 ---
LOCALE = "en-US"                       # 针对你们数据，按需改
MAX_SPEAKERS = 8                       # 期望的最大说话人数量（Azure会自动估计）
POLL_SECONDS = 5                       # 轮询间隔
TIMEOUT_SECONDS = 60 * 60              # 最长等 60 分钟（长会里建议给够）

def _to_seconds(v) -> float:
    # Azure 批量结果的时间字段常见有两种：ticks(100ns) 或 ISO8601 duration。
    # 这里同时兼容：
    if isinstance(v, (int, float)):
        # ticks -> seconds
        return float(v) / 10_000_000.0
    if isinstance(v, str) and v.startswith("PT"):
        # very simple ISO8601 duration parser: PT#S or PT#.#S
        val = v.replace("PT", "").replace("S", "")
        try:
            return float(val)
        except Exception:
            return 0.0
    return 0.0

def _mask(s: str, head=3):
    return (s or "")[:head] + "****"

def _upload_with_sas(local_wav: Path) -> str:
    """
    把音频上传到 Blob，返回带 SAS 的 blob 下载 URL（供 Azure STT 使用）。
    """
    conn_str = app_config.azure_storage.connection_string
    container = app_config.azure_storage.container
    bsc = BlobServiceClient.from_connection_string(conn_str)

    # 容器存在性
    try:
        bsc.create_container(container)
    except Exception:
        pass

    blob_name = f"pipeline/{uuid.uuid4().hex}/{local_wav.name}"
    blob = bsc.get_blob_client(container=container, blob=blob_name)
    logger.info(f"☁️  上传到 Blob：{container}/{blob_name}")
    with open(local_wav, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    # 生成只读 SAS
    sas = generate_blob_sas(
        account_name=blob.account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=bsc.credential.account_key,      # 从连接串里拿到的 key
        permission=BlobSasPermissions(read=True),
        expiry=time.time() + 3600 * 6,               # 6 小时有效
    )
    return f"https://{blob.account_name}.blob.core.windows.net/{container}/{blob_name}?{sas}"

def _azure_batch_transcribe(audio_url: str) -> List[Dict]:
    """
    调 Azure v3.1 批量转写接口（开启 diarization），返回带 speaker 的行。
    """
    region = app_config.azure_speech.service_region
    key    = app_config.azure_speech.speech_key
    base   = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v3.1"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    body = {
        "displayName": f"pipeline-{uuid.uuid4().hex[:8]}",
        "locale": LOCALE,
        "contentUrls": [audio_url],
        "properties": {
            "diarizationEnabled": True,
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked",
            # 新接口里可以不显式给出人数，交给服务自动估计；如需限制可给：{"minCount": 2, "maxCount": MAX_SPEAKERS}
        },
    }

    logger.info(f"☁️  Azure Batch STT 提交：region='{region}', key='{_mask(key)}'")
    with httpx.Client(timeout=None) as cli:
        # 1) 创建任务
        r = cli.post(f"{base}/transcriptions", headers=headers, json=body)
        r.raise_for_status()
        trans_id = r.json()["self"].split("/")[-1]
        logger.info(f"📝  job id = {trans_id}")

        # 2) 轮询
        t0 = time.time()
        while True:
            s = cli.get(f"{base}/transcriptions/{trans_id}", headers=headers).json()
            status = s.get("status")
            if status in {"Succeeded", "Failed"}:
                logger.info(f"🎯  job {trans_id} => {status}")
                if status == "Failed":
                    raise RuntimeError(json.dumps(s.get("errors", []), ensure_ascii=False))
                break
            if time.time() - t0 > TIMEOUT_SECONDS:
                raise TimeoutError("Azure batch transcription timeout.")
            time.sleep(POLL_SECONDS)

        # 3) 列出文件，拿结果 JSON
        files = cli.get(f"{base}/transcriptions/{trans_id}/files", headers=headers).json()
        # 结果文件通常 kind == "Transcription"
        result_files = [f for f in files.get("values", []) if f.get("kind") == "Transcription"]
        if not result_files:
            # 兼容不同字段名
            result_files = files.get("values", [])
        if not result_files:
            raise RuntimeError("No transcription result files returned.")

        content_url = result_files[0]["links"]["contentUrl"]
        res = cli.get(content_url).json()

    # 4) 解析 —— 兼容 recognizedPhrases / combinedRecognizedPhrases 两种结构
    lines: List[Dict] = []
    phrases = res.get("combinedRecognizedPhrases") or res.get("recognizedPhrases") or []
    for p in phrases:
        # 文本
        text = p.get("display") or p.get("lexical") or ""
        if not text and p.get("nBest"):
            text = p["nBest"][0].get("display") or p["nBest"][0].get("lexical") or ""

        # 时间
        start = _to_seconds(p.get("offset") or p.get("startTime") or 0)
        dur   = _to_seconds(p.get("duration") or p.get("durationInTicks") or 0)
        end   = start + dur

        # 说话人
        spk = p.get("speaker") or p.get("speakerNumber") or p.get("speakerId") or "Speaker?"

        if text:
            lines.append({"start": start, "end": end, "text": text, "speaker": str(spk)})

    logger.info(f"📄  Azure 返回 {len(lines)} 行文本（含说话人标签）")
    return sorted(lines, key=lambda x: x["start"])


class SpeakerDiarizationPipeline:
    def process(self, info: ProcessInformation):
        # 如果上游已经含有 speaker（例如重复调用），直接跳过
        if getattr(info, "transcription", None):
            with_speaker = sum(1 for ln in info.transcription if "speaker" in ln) >= max(1, len(info.transcription)//3)
            if with_speaker:
                logger.info("S03: 已含 speaker，跳过说话人分离。")
                info.diarization = info.transcription
                return

        if not info.input_file:
            logger.warning("S03: 缺少 input_file。")
            return

        if USE_AZURE_BATCH:
            logger.info("S03: 使用 Azure Batch（多 Job 并行）进行转写+说话人分离 …")
            lines = azure_batch_diarize(
                input_wav_path=info.input_file,
                segment_seconds=int(os.getenv("SEGMENT_SECONDS", "300")),  # 分片大一些减少 Job 数
                locale=os.getenv("LOCALE", "en-US"),
                parallel_jobs=True,                 # ← 真并行：每分片一个 Job
                max_workers=int(os.getenv("MAX_JOBS", "4")),
            )
            if lines:
                info.transcription = lines
                info.diarization = lines
                logger.info(f"S03: 完成，行数={len(lines)}")
            else:
                logger.warning("S03: Azure 批量返回空结果。")
            return

        # （可选）若你未来想加本地 pyannote 兜底，可在此处补充
        logger.info("S03: 未启用 Azure Batch，跳过。")

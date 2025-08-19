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

USE_AZURE_BATCH: bool = True
LOCALE = "en-US"
MAX_SPEAKERS = 8
POLL_SECONDS = 5
TIMEOUT_SECONDS = 60 * 60

def _to_seconds(v) -> float:
    if isinstance(v, (int, float)):
        return float(v) / 10_000_000.0
    if isinstance(v, str) and v.startswith("PT"):
        val = v.replace("PT", "").replace("S", "")
        try:
            return float(val)
        except Exception:
            return 0.0
    return 0.0

def _mask(s: str, head=3):
    return (s or "")[:head] + "****"

def _upload_with_sas(local_wav: Path) -> str:
    conn_str = app_config.azure_storage.connection_string
    container = app_config.azure_storage.container
    bsc = BlobServiceClient.from_connection_string(conn_str)
    try:
        bsc.create_container(container)
    except Exception:
        pass

    blob_name = f"pipeline/{uuid.uuid4().hex}/{local_wav.name}"
    blob = bsc.get_blob_client(container=container, blob=blob_name)
    logger.info(f"Upload to Blob:{container}/{blob_name}")
    with open(local_wav, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    sas = generate_blob_sas(
        account_name=blob.account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=bsc.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=time.time() + 3600 * 6,
    )
    return f"https://{blob.account_name}.blob.core.windows.net/{container}/{blob_name}?{sas}"

def _azure_batch_transcribe(audio_url: str) -> List[Dict]:
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
        },
    }

    logger.info(f"Azure Batch STT submit: region='{region}', key='{_mask(key)}'")
    with httpx.Client(timeout=None) as cli:
        r = cli.post(f"{base}/transcriptions", headers=headers, json=body)
        r.raise_for_status()
        trans_id = r.json()["self"].split("/")[-1]
        logger.info(f"job id = {trans_id}")

        t0 = time.time()
        while True:
            s = cli.get(f"{base}/transcriptions/{trans_id}", headers=headers).json()
            status = s.get("status")
            if status in {"Succeeded", "Failed"}:
                logger.info(f"job {trans_id} => {status}")
                if status == "Failed":
                    raise RuntimeError(json.dumps(s.get("errors", []), ensure_ascii=False))
                break
            if time.time() - t0 > TIMEOUT_SECONDS:
                raise TimeoutError("Azure batch transcription timeout.")
            time.sleep(POLL_SECONDS)

        files = cli.get(f"{base}/transcriptions/{trans_id}/files", headers=headers).json()
        result_files = [f for f in files.get("values", []) if f.get("kind") == "Transcription"]
        if not result_files:
            result_files = files.get("values", [])
        if not result_files:
            raise RuntimeError("No transcription result files returned.")

        content_url = result_files[0]["links"]["contentUrl"]
        res = cli.get(content_url).json()

    lines: List[Dict] = []
    phrases = res.get("combinedRecognizedPhrases") or res.get("recognizedPhrases") or []
    for p in phrases:
        text = p.get("display") or p.get("lexical") or ""
        if not text and p.get("nBest"):
            text = p["nBest"][0].get("display") or p["nBest"][0].get("lexical") or ""

        start = _to_seconds(p.get("offset") or p.get("startTime") or 0)
        dur   = _to_seconds(p.get("duration") or p.get("durationInTicks") or 0)
        end   = start + dur
        spk = p.get("speaker") or p.get("speakerNumber") or p.get("speakerId") or "Speaker?"

        if text:
            lines.append({"start": start, "end": end, "text": text, "speaker": str(spk)})

    logger.info(f"Azure returned {len(lines)} lines with speaker labels")
    return sorted(lines, key=lambda x: x["start"])


class SpeakerDiarizationPipeline:
    def process(self, info: ProcessInformation):
        if getattr(info, "transcription", None):
            with_speaker = sum(1 for ln in info.transcription if "speaker" in ln) >= max(1, len(info.transcription)//3)
            if with_speaker:
                logger.info("s02: Already contains speaker info, skipping diarization.")
                info.diarization = info.transcription
                return

        if not info.input_file:
            logger.warning("s02: Missing input_file")
            return

        if USE_AZURE_BATCH:
            logger.info("s02: Using Azure Batch (multi-job parallel) for transcription + diarization ")
            lines = azure_batch_diarize(
                input_wav_path=info.input_file,
                segment_seconds=int(os.getenv("SEGMENT_SECONDS", "300")),
                locale=os.getenv("LOCALE", "en-US"),
                parallel_jobs=True,
                max_workers=int(os.getenv("MAX_JOBS", "4")),
            )
            if lines:
                info.transcription = lines
                info.diarization = lines
                logger.info(f"s02: Completed, total lines={len(lines)}")
            else:
                logger.warning("s02: Azure batch returned empty results.")
            return

        logger.info("s02: Azure Batch not enabled, skipping.")

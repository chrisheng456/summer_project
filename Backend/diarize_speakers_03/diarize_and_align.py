"""
Batch audio upload and transcription using Azure Speech v3.1.
Produces an output JSON of the form {"lines": [...]}.

Workflow:
- Convert input audio to denoised 16kHz mono WAV
- Split into fixed-length segments
- Upload segments to Azure Blob with SAS URLs
- Submit all segment URLs to Azure batch transcription (runs in parallel)
- Collect results and merge into unified transcript with {start, end, text, speaker}

Required environment variables:
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

# ---------- Basic setup ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SPEECH_KEY   = os.environ["AZURE_SPEECH_KEY"]
REGION       = os.environ["AZURE_REGION"]
CONN_STR     = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER    = os.environ["AZURE_STORAGE_CONTAINER"]

INPUT_FILE   = os.getenv("INPUT_FILE", "Trustee Meeting Recording (30 June 2025) V1.m4a")
SEG_SECONDS  = int(os.getenv("SEGMENT_SECONDS", "180"))  # default: 3 minutes
LOCALE       = os.getenv("LOCALE", "en-US")
MAX_WORKERS  = int(os.getenv("MAX_UPLOAD_WORKERS", "8"))

OUTPUT_FILE  = "result.json"

SESSION = requests.Session()
SESSION.headers.update({
    "Ocp-Apim-Subscription-Key": SPEECH_KEY,
    "Content-Type": "application/json"
})
TRANSCRIPTIONS_ENDPOINT = f"https://{REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"


# ---------- Utility functions ----------
def run_ffmpeg(cmd, title):
    logging.info(title)
    subprocess.run(cmd, check=True)

def ticks_to_seconds(x):
    """Convert Azure tick values (100ns) into seconds."""
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
    """Remove query parameters from a URL."""
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def norm_speaker(spk):
    """Normalize speaker IDs into SPEAKER_## format."""
    if spk is None:
        return "SPEAKER_00"
    if isinstance(spk, str) and spk.startswith("SPEAKER_"):
        return spk
    try:
        num = int(spk)
        return f"SPEAKER_{num:02d}"
    except Exception:
        return "SPEAKER_00"


# ---------- Blob setup ----------
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
    raise RuntimeError("AccountKey not found in AZURE_STORAGE_CONNECTION_STRING.")

BLOB_BASE = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER}"

def upload_with_read_sas(local_path: str, blob_name: str, hours=6) -> str:
    """Upload a file to Blob storage and return a read-only SAS URL."""
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


# ---------- Main pipeline ----------
def main():
    with tempfile.TemporaryDirectory(prefix="asr_tmp_") as tmpdir:
        denoised = os.path.join(tmpdir, "denoised.wav")
        seg_pattern = os.path.join(tmpdir, "seg_%03d.wav")

        # 1) Convert to WAV (16kHz mono)
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", INPUT_FILE,
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            denoised
        ], " Converting to denoised WAV...")

        # 2) Split audio into fixed-length segments
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", denoised,
            "-f", "segment",
            "-segment_time", str(SEG_SECONDS),
            "-c", "copy",
            seg_pattern
        ], f" Splitting audio into {SEG_SECONDS//60} min segments...")

        segments = sorted(glob.glob(os.path.join(tmpdir, "seg_*.wav")))
        logging.info(f" Segmentation complete, {len(segments)} segments created")

        # 3) Upload segments in parallel with SAS generation
        logging.info(" Uploading segments to Azure Blob with SAS...")

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

        logging.info(f" Uploaded {len(content_urls)} segments")

        # 4) Submit Azure batch transcription job
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
        logging.info(f" Job URL: {trans_url}")

        # 5) Poll job status every 60s
        while True:
            time.sleep(60)
            st = SESSION.get(trans_url, timeout=60).json()
            status = st.get("status")
            logging.info(f"Status: {status}")
            if status in ("Succeeded", "Failed"):
                if status != "Succeeded":
                    raise RuntimeError("Transcription failed")
                break

        # 6) Collect transcription result files
        files_url = st.get("links", {}).get("files")
        files_list = SESSION.get(files_url, timeout=60).json().get("values", [])
        trans_files = [f for f in files_list if f.get("kind") == "Transcription"]
        if not trans_files:
            raise RuntimeError("No transcription result files found")

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

                # Parse timing info
                if "offsetMilliseconds" in p and "durationMilliseconds" in p:
                    start_s = float(p["offsetMilliseconds"]) / 1000.0
                    dur_s   = float(p["durationMilliseconds"]) / 1000.0
                else:
                    start_s = ticks_to_seconds(p.get("offsetInTicks"))
                    dur_s   = ticks_to_seconds(p.get("durationInTicks"))

                # Adjust relative timestamps based on segment index
                if src_url and start_s < SEG_SECONDS + 5:
                    base = (seg_idx or 0) * SEG_SECONDS
                    start_s += base

                lines.append({
                    "start": round(start_s, 2),
                    "end": round(start_s + dur_s, 2),
                    "text": text,
                    "speaker": norm_speaker(p.get("speaker") or p.get("speakerId")),
                })

        # 7) Save final merged transcript
        lines.sort(key=lambda x: (x["start"], x["end"]))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)

        logging.info(f" Completed: {len(lines)} lines saved to {OUTPUT_FILE}")
        # Temporary directory will be auto-deleted when exiting the with-block


if __name__ == "__main__":
    main()

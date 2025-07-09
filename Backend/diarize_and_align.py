#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# 加载 .env 中的 HUGGINGFACE_TOKEN（如果有的话）
load_dotenv()

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
AZURE_JSON  = Path("Trustee Meeting Recording (30 June 2025) V1.json")
INPUT_AUDIO = Path("Trustee Meeting Recording (30 June 2025) V1.m4a")
WAV_PATH    = INPUT_AUDIO.with_suffix(".wav")
OUTPUT_JSON = AZURE_JSON.with_name(AZURE_JSON.stem + "_V2_with_speakers.json")
HF_TOKEN    = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN"))
# ────────────────────────────────────────────────────────────────────────────────

# Step 1: Read transcript
print(f"1) Reading transcript: {AZURE_JSON.name}")
with open(AZURE_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
lines = data.get("lines", [])
print(f"   → {len(lines)} lines loaded.\n", flush=True)

# Step 2: Convert to WAV quietly
print(f"2) Converting audio to WAV: {INPUT_AUDIO.name} → {WAV_PATH.name}", flush=True)
subprocess.run([
    "ffmpeg",
    "-hide_banner", "-loglevel", "error",  # suppress ffmpeg info
    "-y",
    "-i", str(INPUT_AUDIO),
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(WAV_PATH)
], check=True)
print("   → Conversion done.\n", flush=True)

# Step 3: Diarization
print("3) Running pyannote speaker diarization...", flush=True)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=HF_TOKEN
)
diarization = pipeline(str(WAV_PATH))

segments = []
print("   → Segments found:", flush=True)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    seg = {"start": turn.start, "end": turn.end, "speaker": speaker}
    segments.append(seg)
    print(f"     • [{turn.start:.2f}s - {turn.end:.2f}s] → Speaker {speaker}", flush=True)
print(f"   → {len(segments)} segments.\n", flush=True)

# Step 4: Define assign_speaker
def assign_speaker(t0: float, t1: float) -> str:
    overlaps = []
    for s in segments:
        ist = max(t0, s["start"])
        ied = min(t1, s["end"])
        ov = max(0.0, ied - ist)
        if ov > 0:
            overlaps.append((ov, s["speaker"]))
    return max(overlaps, key=lambda x: x[0])[1] if overlaps else "Unknown"

# Step 5: Assign & print merged JSON live
print("4) Assigning speakers and streaming JSON:", flush=True)
for ln in lines:
    t0, t1 = ln.get("start", 0.0), ln.get("end", 0.0)
    ln["speaker"] = assign_speaker(t0, t1)
    print(json.dumps(ln, ensure_ascii=False), flush=True)

print(f"\n   → Assigned speakers to {len(lines)} lines.\n", flush=True)

# Step 6: Write output JSON
print(f"5) Writing output JSON: {OUTPUT_JSON.name}", flush=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"lines": lines}, f, ensure_ascii=False, indent=2)
print("   → Done.", flush=True)

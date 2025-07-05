#!/usr/bin/env python3
import os
import json
import argparse
import tempfile
import subprocess

import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline

HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"

def ensure_wav_pcm16(input_path: str) -> str:
    """
    如果输入已经是 PCM16 单声道 WAV，则直接返回原路径；
    否则用 ffmpeg 转码到临时文件并返回新路径。
    """
    # 检查文件后缀
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        # 再用 torchaudio 看看是不是 16kHz 单声道 pcm_s16le
        try:
            info = torchaudio.info(input_path)
            fmt = info.to_dict().get("format", "")
            if info.num_channels == 1 and info.sample_rate == 16000:
                return input_path
        except Exception:
            pass  # fall through to transcode

    # 转成 WAV PCM16LE, 16kHz, 单声道
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def diarize_and_transcribe(wav_path: str, output_json: str, hf_token: str):
    # 如果格式不符，先转码
    real_wav = ensure_wav_pcm16(wav_path)

    print("▶ Running speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=hf_token
    )
    diarization = pipeline(real_wav)

    print("▶ Loading Whisper ASR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("small", device=device)

    print("▶ Loading audio and slicing by speaker segments...")
    waveform, sample_rate = torchaudio.load(real_wav)
    waveform = waveform.to(device)

    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = segment.start, segment.end
        sf, ef = int(start * sample_rate), int(end * sample_rate)
        clip = waveform[:, sf:ef]

        # 临时存片段去识别
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, clip.cpu(), sample_rate)
            result = asr_model.transcribe(tmp.name, fp16=(device=="cuda"))
        os.unlink(tmp.name)

        text = result.get("text", "").strip()
        segments.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end":   round(end, 3),
            "text":  text
        })

    # 排序并写入 JSON
    segments.sort(key=lambda x: x["start"])
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"✅ Aligned JSON saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step diarization + ASR: any audio → aligned JSON"
    )
    parser.add_argument(
        "wav",
        help="输入音频文件路径（支持 mp3/m4a/flac/ogg/… 自动转码）"
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv(HF_TOKEN_ENV),
        help=f"HuggingFace 访问令牌，或设置环境变量 {HF_TOKEN_ENV}"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        parser.error(f"找不到音频文件: {args.wav}")
    if not args.hf_token:
        parser.error(f"请通过 --hf_token 或环境变量 {HF_TOKEN_ENV} 提供 HuggingFace 访问令牌。")

    # 输出文件名：脚本同目录，基于输入名 + _aligned.json
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.splitext(os.path.basename(args.wav))[0] + "_aligned.json"
    output_path = os.path.join(script_dir, base)

    diarize_and_transcribe(args.wav, output_path, args.hf_token)
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk

from ...schema.process_information import ProcessInformation
from ...config import app_config

# 切块参数：每块 5 分钟，块间重叠 1 秒（避免句子被截断）
CHUNK_SECONDS: float = 300.0
CHUNK_OVERLAP: float = 1.0



def _build_speech_config() -> speechsdk.SpeechConfig:
    """
    统一读取 key/region，并把使用到的值打印出来（key 打码）。
    读取优先级：系统环境变量 > .env（app_config）
    """
    key = os.getenv("AZURE_SPEECH_KEY") or app_config.azure_speech.speech_key
    region = os.getenv("AZURE_SPEECH_REGION") or app_config.azure_speech.service_region

    # 打码显示，避免泄露完整 key
    if key:
        masked = f"{key[:4]}***{key[-4:]}" if len(key) >= 10 else (key[:4] + "***")
    else:
        masked = "None"

    logger.info(f"🔐 Azure STT 使用配置：region='{region}', key='{masked}'")
    if not key or not region:
        raise RuntimeError(
            f"AZURE_SPEECH_KEY/AZURE_SPEECH_REGION 缺失：key={'set' if key else 'missing'}, region={region!r}"
        )

    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    # 和你们原来的配置保持一致
    try:
        cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
        cfg.request_word_level_timestamps()
    except Exception:
        pass
    return cfg


class SpeechToTextPipeline:
    def process(self, data: ProcessInformation) -> None:
        """
        把 data.input_file（已由 s00 转为 16k 单声道 WAV）做切块识别，
        最终把识别结果写回 data.transcription（list[dict]）。
        """
        input_path = Path(data.input_file)
        audio = AudioSegment.from_file(input_path)
        duration_s = len(audio) / 1000.0
        logger.info(f"🗣️ 待识别音频：{input_path.name}，时长 {duration_s/60:.2f} 分钟")

        # 生成切块时间段
        chunks: List[tuple[float, float]] = []
        start = 0.0
        while start < duration_s:
            end = min(start + CHUNK_SECONDS, duration_s)
            chunks.append((start, end))
            if end >= duration_s:
                break
            start = end - CHUNK_OVERLAP

        logger.info(f"🔪 切块数：{len(chunks)}（每块 {CHUNK_SECONDS/60:.0f} 分钟，重叠 {CHUNK_OVERLAP} 秒）")

        # 临时块目录
        chunk_dir = Path(data.tmp_dir) / "s01_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # 构建一次 speech_config（可复用）
        speech_config = _build_speech_config()

        all_lines: List[Dict[str, Optional[float] | str]] = []

        for idx, (st, ed) in enumerate(chunks, start=1):
            # 导出当前块
            start_ms = int(st * 1000)
            dur_ms = int((ed - st) * 1000)
            chunk_audio = audio[start_ms:start_ms + dur_ms]
            chunk_path = chunk_dir / f"chunk_{idx:03d}.wav"
            logger.info(f"▶ 第 {idx}/{len(chunks)} 块导出：[{st:.2f}s ~ {ed:.2f}s] → {chunk_path.name}")
            chunk_audio.export(chunk_path, format="wav")

            audio_config = speechsdk.AudioConfig(filename=str(chunk_path))
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            lines_this_chunk: List[Dict[str, Optional[float] | str]] = []
            is_done = False

            # 识别结果（逐句）
            def _on_transcribed(evt):
                nonlocal lines_this_chunk
                res = evt.result
                if res.reason == speechsdk.ResultReason.RecognizedSpeech and res.text.strip():
                    # offset/duration 单位是 100ns ticks，换算为秒；加上块起点得到全局时间
                    start_sec = getattr(res, "offset", None)
                    dur_sec = getattr(res, "duration", None)
                    try:
                        start_sec = st + (start_sec / 10_000_000) if start_sec is not None else None
                        dur_sec = (dur_sec / 10_000_000) if dur_sec is not None else None
                    except Exception:
                        start_sec = None
                        dur_sec = None

                    lines_this_chunk.append({
                        "start": float(start_sec) if start_sec is not None else None,
                        "dur": float(dur_sec) if dur_sec is not None else None,
                        "text": res.text,
                    })

            # 块结束
            def _on_session_stopped(evt):
                nonlocal is_done
                logger.info("=== 本块识别结束 ===")
                is_done = True

            # 取消/错误
            def _on_canceled(evt):
                nonlocal is_done
                details = speechsdk.CancellationDetails(evt)
                logger.error(
                    f"CANCELED: {details.reason} / {details.error_details} "
                    f"(SessionId: {getattr(evt, 'session_id', 'n/a')})"
                )
                is_done = True

            recognizer.recognized.connect(_on_transcribed)
            recognizer.session_stopped.connect(_on_session_stopped)
            recognizer.canceled.connect(_on_canceled)

            logger.info(f"🎧 开始识别：{chunk_path.name}（第 {idx} 块，{(ed-st)/60:.2f} 分钟）")
            recognizer.start_continuous_recognition_async()
            while not is_done:
                time.sleep(0.1)
            recognizer.stop_continuous_recognition_async()

            all_lines.extend(lines_this_chunk)
            logger.info(f"✅ 第 {idx} 块识别完成，新增 {len(lines_this_chunk)} 句。")

        # 汇总排序（尽量按 start 排；缺失 start 的置后）
        all_lines.sort(key=lambda x: (float('inf') if x.get("start") is None else x["start"]))  # type: ignore
        data.transcription = all_lines
        logger.info(f"🎉 全部完成：共 {len(all_lines)} 句。")

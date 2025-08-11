from loguru import logger
from typing import List, Dict, Optional
import time
import azure.cognitiveservices.speech as speechsdk

from app.schema.process_information import ProcessInformation
from app.config import app_config


class SpeechToTextPipeline:

    def process(self, data: ProcessInformation):
        """
        处理音频文件，返回逐字稿数据（含时间戳）。
        """

        # 1. SpeechConfig & 功能开关
        speech_config = speechsdk.SpeechConfig(
            subscription=app_config.azure_speech.speech_key,
            region=app_config.azure_speech.service_region,
        )

        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
            "TrueText",
        )

        speech_config.request_word_level_timestamps()

        # 2. 准备音频文件
        audio_config = speechsdk.AudioConfig(filename=data.input_file)

        # 3. 创建转写器
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config, audio_config=audio_config
        )

        lines: List[Dict[str, Optional[float]]] = []  # 存储每句的时间戳和文本
        is_done = False

        def _on_transcribed(evt: speechsdk.SpeechRecognitionEventArgs):
            if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            text = evt.result.text.strip()
            if not text:
                return
            start_sec = evt.result.offset / 10_000_000
            end_sec = start_sec + (evt.result.duration / 10_000_000)
            lines.append({"start": start_sec, "end": end_sec, "text": text})
            logger.info(f"[{start_sec:.2f}s - {end_sec:.2f}s] {text}")

        def _on_session_stopped(_):
            logger.info("=== 识别结束 ===")
            nonlocal is_done
            is_done = True

        def _on_canceled(evt):
            details = speechsdk.CancellationDetails(evt)
            logger.error(
                f"CANCELED: {details.reason} / {details.error_details}"
            )
            nonlocal is_done
            is_done = True

        transcriber.transcribed.connect(_on_transcribed)
        transcriber.session_stopped.connect(_on_session_stopped)
        transcriber.canceled.connect(_on_canceled)

        # 5. 开始转写 & 等待结束
        logger.info(f"▶ 开始识别 {data.input_file} ...")
        transcriber.start_transcribing_async()
        while not is_done:
            time.sleep(0.1)
        transcriber.stop_transcribing_async()

        # 6. 整理 & 保存输出（不加时间戳，使用输入文件名前缀）
        lines.sort(key=lambda x: x["start"])

        data.transcription = lines

        logger.info(f"✅ 识别完成，共 {len(lines)} 句。逐字稿已保存。")

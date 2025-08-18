# Backend/api/app/pipeline/__init__.py
from __future__ import annotations
import tempfile
from .s04_customer_api import CustomerApiPipeline
from .s00_audio_converter import AudioConverterPipeline
from .s03_speaker_diarization import SpeakerDiarizationPipeline
from .s02_data_cleaning import DataCleaningPipeline
from .s04_agenda_segmenter import AgendaSegmenterPipeline
from .s05_text_classification import TextClassificationPipeline
from .s06_text_summary import TextSummaryPipeline
from ..schema.process_information import ProcessInformation

def _compact_info(info: ProcessInformation) -> None:
    """丢弃中间产物，内存与响应都更干净。"""
    info.transcription = None
    info.cleaned_transcription = None
    info.diarization = None
    info.speaker_segments = None
    info.classified_lines = None
    # 如果不需要聚合的 summary 字段，可一并清理（保留 agenda 内的 summary）：
    # info.summary = None

def process_pipeline(
    input_file_content: bytes,
    scheme_id: str,
    meeting_id: str,
    bearer_token: str,
) -> ProcessInformation:
    """
    精简版流水线（不保留中间结果）：
    S00 客户API → S00 转 WAV → S03 Azure Batch(并行,含转写+说话人)
      → S02 清洗 → S04 议程对齐 → S05 分类 → S06 摘要
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        info = ProcessInformation(
            tmp_dir=tmp_dir,
            scheme_id=scheme_id,
            meeting_id=meeting_id,
            bearer_token=bearer_token,
        )
        # 写入上传音频
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".wav", delete=False) as f:
            f.write(input_file_content)
            info.input_file = f.name

        # S00 客户 API（meeting detail/agenda）
        CustomerApiPipeline(scheme_id, meeting_id, bearer_token).process(info)

        # S00 转 WAV（稳妥：统一 16kHz mono）
        AudioConverterPipeline().process(info)

        # S03 Azure Batch（多 Job 并行：一次拿转写+说话人）
        SpeakerDiarizationPipeline().process(info)

        # S02 清洗（就地覆盖 text）
        DataCleaningPipeline().process(info)

        # S04 按时间窗口分配到 agenda[*].lines
        AgendaSegmenterPipeline().process(info)

        # S05 分类（按议程）
        TextClassificationPipeline().process(info)

        # S06 摘要（按议程）
        TextSummaryPipeline().process(info)

        # —— 只保留最终版：清理中间字段 —— #
        _compact_info(info)

        return info

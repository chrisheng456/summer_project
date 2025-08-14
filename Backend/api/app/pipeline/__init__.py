from __future__ import annotations
import tempfile
from pathlib import Path

from ..schema.process_information import ProcessInformation
from .s04_customer_api import CustomerApiPipeline
from .s00_audio_converter import AudioConverterPipeline
from .s01_speech_to_text import SpeechToTextPipeline
from .s02_data_cleaning import DataCleaningPipeline
from .s03_speaker_diarization import SpeakerDiarizationPipeline
from .s05_text_classification import TextClassificationPipeline
from .s06_text_summary import TextSummaryPipeline

def process_pipeline(
    input_file_content: bytes,
    scheme_id: str,
    meeting_id: str,
    bearer_token: str,
) -> ProcessInformation:
    with tempfile.TemporaryDirectory() as tmp_dir:
        info = ProcessInformation(tmp_dir=tmp_dir,
                                  scheme_id=scheme_id,
                                  meeting_id=meeting_id,
                                  bearer_token=bearer_token)

        # write uploaded file into tmp dir
        with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as f:
            f.write(input_file_content)
            info.input_file = f.name

        # S00: customer API first
        CustomerApiPipeline(scheme_id, meeting_id, bearer_token).process(info)

        # S01~S06 as before
        AudioConverterPipeline().process(info)
        SpeechToTextPipeline().process(info)
        DataCleaningPipeline().process(info)
        SpeakerDiarizationPipeline().process(info)
        TextClassificationPipeline().process(info)
        TextSummaryPipeline().process(info)

        return info

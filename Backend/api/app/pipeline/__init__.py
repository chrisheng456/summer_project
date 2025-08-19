from __future__ import annotations
import tempfile
from .s00_customer_api import CustomerApiPipeline
from .s01_audio_converter import AudioConverterPipeline
from .s02_transcribe_diarize import SpeakerDiarizationPipeline
from .s03_data_cleaning import DataCleaningPipeline
from .s04_agenda_segmenter import AgendaSegmenterPipeline
from .s05_text_classification import TextClassificationPipeline
from .s06_text_summary import TextSummaryPipeline
from ..schema.process_information import ProcessInformation

def _compact_info(info: ProcessInformation) -> None:
    info.transcription = None
    info.cleaned_transcription = None
    info.diarization = None
    info.speaker_segments = None
    info.classified_lines = None

def process_pipeline(
    input_file_content: bytes,
    scheme_id: str,
    meeting_id: str,
    bearer_token: str,
) -> ProcessInformation:

    with tempfile.TemporaryDirectory() as tmp_dir:
        info = ProcessInformation(
            tmp_dir=tmp_dir,
            scheme_id=scheme_id,
            meeting_id=meeting_id,
            bearer_token=bearer_token,
        )

        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".wav", delete=False) as f:
            f.write(input_file_content)
            info.input_file = f.name

        CustomerApiPipeline(scheme_id, meeting_id, bearer_token).process(info)

        AudioConverterPipeline().process(info)

        SpeakerDiarizationPipeline().process(info)

        DataCleaningPipeline().process(info)

        AgendaSegmenterPipeline().process(info)

        TextClassificationPipeline().process(info)

        TextSummaryPipeline().process(info)

        _compact_info(info)

        return info

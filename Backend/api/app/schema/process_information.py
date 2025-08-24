from __future__ import annotations
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ProcessInformation(BaseModel):
    tmp_dir: str
    input_file: Optional[str] = None

    scheme_id: Optional[str] = None
    meeting_id: Optional[str] = None
    bearer_token: Optional[str] = None
    customer_meeting_detail: Optional[Dict] = None
    transcription: Optional[List[Dict]] = None

    cleaned_transcription: Optional[List[Dict]] = None

    diarization: Optional[List[Dict]] = None
    speaker_segments: Optional[List[Dict]] = None
    classified_lines: Optional[List[Dict]] = None
    label_counts: Optional[Dict[str, int]] = None

    summary: Optional[Dict] = None

    class Config:
        extra = "allow"

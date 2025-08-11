from typing import List
from pydantic import BaseModel


class ProcessInformation(BaseModel):
    # 临时目录
    tmp_dir: str
    input_file: str = None

    # 逐字稿
    transcription: List[dict] = None

    # 说话人分离结果
    customer_meeting_detail: dict = None

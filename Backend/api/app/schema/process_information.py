from __future__ import annotations
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ProcessInformation(BaseModel):
    # 基础运行上下文
    tmp_dir: str
    input_file: Optional[str] = None

    # —— S00: 客户 API ——（meeting detail/agenda 等）
    scheme_id: Optional[str] = None
    meeting_id: Optional[str] = None
    bearer_token: Optional[str] = None
    customer_meeting_detail: Optional[Dict] = None

    # —— S01: 语音转写 ——（逐句）
    # 每条建议结构：{"start": float, "end": float, "text": str, "speaker": Optional[str]}
    transcription: Optional[List[Dict]] = None

    # —— S02: 数据清洗 ——（本次需求：仅就地清洗；如需保留副本则用此字段）
    cleaned_transcription: Optional[List[Dict]] = None

    # —— S03: 说话人分离/对齐结果 ——（若你用本地 pyannote，也可额外产出 segments）
    diarization: Optional[List[Dict]] = None            # or speaker_segments
    speaker_segments: Optional[List[Dict]] = None       # 兼容旧命名

    # —— S04: 按议程切片后的结构 ——（直接写回 meeting_detail.agenda[i].lines）
    # 这里不单独再存一份，统一从 customer_meeting_detail["agenda"] 读取

    # —— S05: 文本分类 ——（如果你还想单独返回一份聚合结果）
    classified_lines: Optional[List[Dict]] = None
    label_counts: Optional[Dict[str, int]] = None

    # —— S06: 摘要 ——（如果你想把每个 agenda 的摘要同时汇总，可放这里）
    summary: Optional[Dict] = None

    class Config:
        # 允许将来临时加字段（防御式，避免再次因为字段名差异挂掉）
        extra = "allow"

import datetime
import json
import tempfile
from loguru import logger
from fastapi import UploadFile

from app.schema.process_information import ProcessInformation
from app.models import ConversionTask

from .s00_audio_converter import AudioConverterPipeline
from .s01_speech_to_text import SpeechToTextPipeline
from .s02_data_cleaning import DataCleaningPipeline
from .s03_speaker_diarization import SpeakerDiarizationPipeline
from .s04_customer_api import CustomerApiPipeline
from .s05_text_classification import TextClassificationPipeline
from .s06_text_summary import TextSummaryPipeline


def process(
    task_id: int,
    file_content: bytes,
    scheme_id: str = None,
    meeting_id: str = None,
):
    """处理转换任务"""
    # 创建 ConversionTask 实例
    try:
        # 任务开始，更新状态
        task = ConversionTask.get(ConversionTask.id == task_id)
        task.status = "processing"
        task.updated_at = datetime.datetime.now()
        task.save()

        # 处理文件
        info = process_pipeline(file_content, scheme_id, meeting_id)

        result = info.customer_meeting_detail or {}

        # 处理成功，写入结果
        task.result_json = json.dumps(result, ensure_ascii=False)
        task.status = "done"
        task.updated_at = datetime.datetime.utcnow()
        task.save()
    except Exception as e:
        logger.error(f"处理任务 {task_id} 时发生错误: {e}", exc_info=True)
        # 处理异常，写入错误信息
        task = ConversionTask.get_or_none(ConversionTask.id == task_id)
        if task:
            task.status = "failed"
            task.error_message = str(e)
            task.updated_at = datetime.datetime.utcnow()
            task.save()


def process_pipeline(
    input_file_content: bytes,
    scheme_id: str = None,
    meeting_id: str = None,
):
    """处理转换管道"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建 ProcessInformation 实例
        # 这里的 tmp_dir 是临时目录路径
        info = ProcessInformation(tmp_dir=tmp_dir)

        # 将上传的文件保存到临时目录
        with tempfile.NamedTemporaryFile(
            dir=tmp_dir, delete=False
        ) as temp_file:
            temp_file.write(input_file_content)
            info.input_file = temp_file.name

            # 0. 音频转换
            AudioConverterPipeline().process(info)
            # 1. 语音转文本
            SpeechToTextPipeline().process(info)
            # 2. 数据清洗
            DataCleaningPipeline().process(info)
            # 3. 说话人分离
            SpeakerDiarizationPipeline().process(info)
            # 4. 客户API补充会议信息
            CustomerApiPipeline(
                scheme_id=scheme_id, meeting_id=meeting_id
            ).process(info)
            # 5. 文本分类
            TextClassificationPipeline().process(info)
            # 6. 文本摘要
            TextSummaryPipeline().process(info)
    return info

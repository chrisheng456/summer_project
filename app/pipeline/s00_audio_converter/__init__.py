from loguru import logger
from pydub import AudioSegment
from pathlib import Path

from app.schema.process_information import ProcessInformation


class AudioConverterPipeline:

    def process(self, data: ProcessInformation) -> None:
        """
        处理音频文件，将其转换为 WAV 格式并返回转换后的文件路径。
        """

        # 1. 获取输入文件路径
        input_file = Path(data.input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"找不到源文件：{input_file.resolve()}")

        # 2. 定义输出文件路径
        output_file = Path(data.tmp_dir) / f"{input_file.stem}.wav"

        # 3. 转换音频格式
        logger.info(
            f"🔄 转换音频：{input_file.name} → {output_file.name} （16kHz 单声道 WAV）"
        )
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")

        data.input_file = str(output_file)

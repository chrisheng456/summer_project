from loguru import logger
from pydub import AudioSegment
from pathlib import Path

from ...schema.process_information import ProcessInformation


class AudioConverterPipeline:

    def process(self, data: ProcessInformation) -> None:

        input_file = Path(data.input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"The source file cannot be found:{input_file.resolve()}")
        output_file = Path(data.tmp_dir) / f"{input_file.stem}.wav"

        logger.info(
            f"Audio:{input_file.name} → {output_file.name} (16kHz mono WAV)"
        )
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")

        data.input_file = str(output_file)

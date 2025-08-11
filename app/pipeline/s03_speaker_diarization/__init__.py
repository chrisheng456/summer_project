from pyannote.audio import Pipeline

from app.config import app_config
from app.schema.process_information import ProcessInformation


class SpeakerDiarizationPipeline:
    def process(self, info: ProcessInformation):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=app_config.huggingface.token,
        )
        diarization = pipeline(str(info.input_file))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )

        def assign_speaker(t0, t1):
            overlaps = []
            for s in segments:
                ist = max(t0, s["start"])
                ied = min(t1, s["end"])
                ov = max(0.0, ied - ist)
                if ov > 0:
                    overlaps.append((ov, s["speaker"]))
            return (
                max(overlaps, key=lambda x: x[0])[1] if overlaps else "Unknown"
            )

        if not hasattr(info, "transcription") or not info.transcription:
            return
        for ln in info.transcription:
            t0, t1 = ln.get("start", 0.0), ln.get("end", 0.0)
            ln["speaker"] = assign_speaker(t0, t1)

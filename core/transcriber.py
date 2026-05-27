_fw_model = None
_fw_model_size = None

SAMPLE_RATE = 16000


def _get_fw_model(size="base"):
    global _fw_model, _fw_model_size
    if _fw_model is None or _fw_model_size != size:
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel(size, device="cpu", compute_type="int8")
        _fw_model_size = size
    return _fw_model


def preload(size="base"):
    _get_fw_model(size)


def transcribe(audio_path, model_size="base", diarize=False, hf_token=""):
    """
    Returns (text, words, speaker_runs).
    speaker_runs is a list of {"speaker": "SPEAKER_0", "text": "..."} or None.
    """
    model = _get_fw_model(model_size)
    segments_iter, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    segments = []
    all_words = []
    for seg in segments_iter:
        seg_words = []
        for w in (seg.words or []):
            wd = {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            seg_words.append(wd)
            all_words.append(wd)
        segments.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
            "words": seg_words,
        })

    if diarize and hf_token:
        import torch
        import numpy as np
        import wave as _wave
        from pyannote.audio import Pipeline as PyannotePipeline

        # load audio as float32 tensor — avoids ffmpeg
        with _wave.open(audio_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)  # (1, time)

        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        output = pipeline({"waveform": audio_tensor, "sample_rate": SAMPLE_RATE})

        # newer pyannote wraps result in DiarizeOutput
        annotation = getattr(output, "speaker_diarization", output)

        # build speaker ranges from pyannote Annotation
        speaker_ranges = [
            (seg.start, seg.end, label)
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]

        def speaker_at(t):
            for start, end, lbl in speaker_ranges:
                if start <= t <= end:
                    return lbl
            if not speaker_ranges:
                return "SPEAKER_0"
            return min(speaker_ranges, key=lambda r: abs((r[0] + r[1]) / 2 - t))[2]

        # assign speaker to each segment
        runs = []
        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            spk = speaker_at(mid)
            if runs and runs[-1]["speaker"] == spk:
                runs[-1]["text"] += " " + seg["text"]
            else:
                runs.append({"speaker": spk, "text": seg["text"]})

        text = "\n".join(f"{r['speaker']}: {r['text']}" for r in runs)
        return text, all_words, runs

    text = " ".join(s["text"] for s in segments)
    return text, all_words, None

_model = None
_model_size = None


def _load(size="base"):
    global _model, _model_size
    if _model is None or _model_size != size:
        from faster_whisper import WhisperModel
        _model = WhisperModel(size, device="cpu", compute_type="int8")
        _model_size = size
    return _model


def preload(size="base"):
    _load(size)


def transcribe(audio_path, model_size="base"):
    model = _load(model_size)
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    words = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
    return " ".join(text_parts), words

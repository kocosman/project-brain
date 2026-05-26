_model = None
_model_size = None
_align_models = {}  # language -> (model_a, metadata)

DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def _get_model(size="base"):
    global _model, _model_size
    import whisperx
    if _model is None or _model_size != size:
        _model = whisperx.load_model(size, device=DEVICE, compute_type=COMPUTE_TYPE)
        _model_size = size
    return _model


def _get_aligner(language):
    import whisperx
    if language not in _align_models:
        model_a, meta = whisperx.load_align_model(language_code=language, device=DEVICE)
        _align_models[language] = (model_a, meta)
    return _align_models[language]


def preload(size="base"):
    _get_model(size)


def transcribe(audio_path, model_size="base", num_speakers=1, hf_token=""):
    """
    Returns (text, words, speaker_runs).
    speaker_runs is a list of {"speaker": "SPEAKER_00", "text": "..."} when
    diarization ran, else None.
    """
    import whisperx

    model = _get_model(model_size)
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=4)
    language = result.get("language", "en")

    # align for word-level timestamps
    try:
        model_a, meta = _get_aligner(language)
        result = whisperx.align(
            result["segments"], model_a, meta, audio,
            device=DEVICE, return_char_alignments=False,
        )
    except Exception:
        pass

    segments = result.get("segments", [])

    # diarize when multiple speakers and token provided
    if num_speakers >= 2 and hf_token:
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=DEVICE
            )
            diarize_segs = diarize_model(
                audio, min_speakers=num_speakers, max_speakers=num_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segs, result)
            segments = result.get("segments", [])
            return _build_diarized(segments)
        except Exception:
            pass  # fall through to plain

    text = " ".join(s["text"].strip() for s in segments)
    words = _extract_words(segments)
    return text, words, None


def _build_diarized(segments):
    runs = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_?")
        text = seg["text"].strip()
        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["text"] += " " + text
        else:
            runs.append({"speaker": speaker, "text": text})

    text = "\n".join(f"{r['speaker']}: {r['text']}" for r in runs)
    words = _extract_words(segments)
    return text, words, runs


def _extract_words(segments):
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": w.get("start", 0),
                "end": w.get("end", 0),
                "probability": w.get("score", 1.0),
            })
    return words

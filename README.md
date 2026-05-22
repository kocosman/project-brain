# Project Brain — Local Meeting Transcriber

A fully local Windows desktop app for recording, transcribing, and storing meeting notes.  
No cloud APIs. All AI runs on-device via [faster-whisper](https://github.com/guillaumekm/faster-whisper).

## Features

- **Record** meetings directly in the app
- **Transcribe** audio locally using Whisper (CPU, no GPU required)
- **Save** raw transcripts and summaries to separate project folders
- Fully offline — no API keys, no internet required

## Requirements

- Python 3.11+
- `pip install faster-whisper sounddevice numpy`

## How to Run

```bash
python whisper_recorder.py
```

## Folder Structure

```
whisper_recorder/
  whisper_recorder.py   # main app
  .gitignore
  README.md
projects/               # created at runtime, not tracked by git
  [project-name]/
    [YYYY-MM-DD_name]/
      transcript.txt
      summary.md
```

## Whisper Model Sizes

| Model  | Size  | Speed  | Accuracy |
|--------|-------|--------|----------|
| tiny   | 75 MB | fast   | lower    |
| base   | 145 MB| medium | good     |
| small  | 466 MB| slower | better   |
| medium | 1.5 GB| slow   | best     |

Change `MODEL_SIZE` at the top of `whisper_recorder.py` to switch models.

## Roadmap

- [ ] Ollama integration for local LLM summaries
- [ ] Project memory — ask questions across all meeting notes
- [ ] Export to PDF

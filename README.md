# Project Brain — Local AI Meeting Transcriber

A fully local Windows desktop app for recording, transcribing, summarizing, and querying meeting notes.  
No cloud APIs. No API keys. All AI runs on-device.

## Features

| Feature | How |
|---|---|
| Record meetings | sounddevice |
| Transcribe speech | faster-whisper (CPU, offline) |
| Summarize notes | Ollama (local LLM) |
| Ask questions across meetings | Ollama Q&A over all summaries |
| Search meetings | Full-text search across all projects |
| Export PDF | markdown2 + weasyprint |

## Requirements

**Python 3.11+**

```bash
pip install customtkinter faster-whisper sounddevice numpy requests markdown2 weasyprint
```

**Ollama** (for summarize / Q&A):
1. Download from https://ollama.com
2. `ollama serve`
3. `ollama pull llama3.2`   ← or any model you prefer

## How to Run

```bash
python main.py
```

On first launch the app checks if Ollama is running. If not, a warning dialog shows — recording and transcription still work without it.

## File Structure

```
main.py                         # entry point
core/
  config.py                     # settings read/write
  llm.py                        # Ollama wrapper
  storage.py                    # flat-file project storage
  transcriber.py                # Whisper wrapper
  summarizer.py                 # per-type summarization prompts
  memory.py                     # search, recap, Q&A
ui/
  recorder_tab.py               # Tab 1: record + transcribe + summarize
  memory_tab.py                 # Tab 2: recap + search + Q&A
  settings_tab.py               # Tab 3: config
settings.json                   # user preferences (git-ignored)
projects/                       # your data (git-ignored)
  [project-name]/
    [YYYY-MM-DD_meeting-name]/
      transcript.txt
      summary.md
      meta.json
```

## Whisper Model Sizes

| Model  | Size   | Notes |
|--------|--------|-------|
| tiny   | 75 MB  | fastest, lower accuracy |
| base   | 145 MB | good balance (default) |
| small  | 466 MB | better accuracy |
| medium | 1.5 GB | best accuracy, slowest |

Change in Settings tab — model downloads automatically on first use.

## Ollama Model Recommendations

| Model | VRAM/RAM | Notes |
|---|---|---|
| llama3.2 | ~4 GB | fast, good quality (recommended) |
| mistral | ~4 GB | good for structured output |
| llama3.1:8b | ~6 GB | stronger reasoning |
| phi3 | ~2 GB | lightweight option |

## Meeting Types

Each type has a tailored summarization prompt:
- **General** — key points, action items, decisions
- **Standup** — done / in-progress / blockers per person
- **Design Review** — problem, solution, concerns, next steps
- **Client Call** — topics, requests, commitments, follow-ups
- **Retrospective** — what went well, improvements, actions
